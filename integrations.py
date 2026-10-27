"""Optional pregame weather and injury ingestion. Both outputs are game_id keyed CSVs.
Weather: Open-Meteo forecast API, no key. Injuries: user/provider supplied injury report CSV.
Never use a postgame observation as if it was available pregame.
"""
from pathlib import Path
import pandas as pd, requests
BASE=Path(__file__).parent/'data';BASE.mkdir(exist_ok=True)
# Stadium coordinates are provided by the user in data/stadiums.csv to avoid geocoding ambiguity.
def fetch_weather():
    games=pd.read_csv(BASE/'games.csv'); stadiums=pd.read_csv(BASE/'stadiums.csv')
    games=games[games.home_score.isna()].copy(); rows=[]
    for _,g in games.iterrows():
        m=stadiums[stadiums.team==g.home_team]
        if m.empty:continue
        s=m.iloc[0]
        if str(s.get('roof','')).lower() in ('dome','closed'):continue
        try:
            r=requests.get('https://api.open-meteo.com/v1/forecast',params={'latitude':s.latitude,'longitude':s.longitude,'hourly':'temperature_2m,precipitation,wind_speed_10m','temperature_unit':'fahrenheit','wind_speed_unit':'mph','timezone':'UTC','forecast_days':16},timeout=20);r.raise_for_status()
            h=pd.DataFrame(r.json()['hourly']); h['time']=pd.to_datetime(h.time,utc=True)
            # Prefer exact UTC kickoff provided by schedule; do not guess a timezone.
            if 'gametime' not in g or pd.isna(g.gametime):continue
            dt=pd.to_datetime(str(g.gameday)+' '+str(g.gametime),utc=True,errors='coerce')
            if pd.isna(dt):continue
            closest=h.iloc[(h.time-dt).abs().argmin()]
            if abs((closest.time-dt).total_seconds())>5400:continue
            rows.append({'game_id':g.game_id,'wind_mph':closest.wind_speed_10m,'temp_f':closest.temperature_2m,'precip_mm':closest.precipitation})
        except (requests.RequestException,ValueError,KeyError) as e:print('Weather unavailable:',g.get('game_id'),e)
    pd.DataFrame(rows,columns=['game_id','wind_mph','temp_f','precip_mm']).to_csv(BASE/'weather.csv',index=False)
    print('Weather rows:',len(rows))

def import_injuries():
    """Normalize injury adjustments from injury_report.csv: game_id,team,points_impact.
    points_impact is user-estimated expected scoring impact; e.g. -2.0 for an absent QB.
    Do not treat these subjective estimates as learned coefficients.
    """
    p=BASE/'injury_report.csv'
    if not p.exists():raise FileNotFoundError('Create data/injury_report.csv from an official NFL injury report; see README')
    report=pd.read_csv(p);games=pd.read_csv(BASE/'games.csv')
    required={'game_id','team','points_impact'}
    if not required.issubset(report):raise ValueError(f'Missing injury report columns {required-set(report)}')
    r=report.groupby(['game_id','team'],as_index=False).points_impact.sum();out=[]
    for _,g in games[games.home_score.isna()].iterrows():
        def value(team):
            m=r[(r.game_id==g.game_id)&(r.team==team)]
            return float(m.points_impact.iloc[0]) if len(m) else 0.
        out.append({'game_id':g.game_id,'home_injury_points':value(g.home_team),'away_injury_points':value(g.away_team)})
    pd.DataFrame(out).to_csv(BASE/'injuries.csv',index=False)
    print('Injury adjustments:',len(out))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['weather','injuries']);args=p.parse_args()
    fetch_weather() if args.action=='weather' else import_injuries()
