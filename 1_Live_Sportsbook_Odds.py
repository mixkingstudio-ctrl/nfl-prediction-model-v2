"""Live US sportsbook odds companion page for NFL Model 5.1.
Reads existing predictions without modifying the model or historical backtests.
"""
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PICKS = ROOT / 'artifacts' / 'sunday_picks.csv'
API = 'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/'
TEAMS = {
 'ARI':'Arizona Cardinals','ATL':'Atlanta Falcons','BAL':'Baltimore Ravens','BUF':'Buffalo Bills',
 'CAR':'Carolina Panthers','CHI':'Chicago Bears','CIN':'Cincinnati Bengals','CLE':'Cleveland Browns',
 'DAL':'Dallas Cowboys','DEN':'Denver Broncos','DET':'Detroit Lions','GB':'Green Bay Packers',
 'HOU':'Houston Texans','IND':'Indianapolis Colts','JAX':'Jacksonville Jaguars','KC':'Kansas City Chiefs',
 'LA':'Los Angeles Rams','LAC':'Los Angeles Chargers','LV':'Las Vegas Raiders','MIA':'Miami Dolphins',
 'MIN':'Minnesota Vikings','NE':'New England Patriots','NO':'New Orleans Saints','NYG':'New York Giants',
 'NYJ':'New York Jets','PHI':'Philadelphia Eagles','PIT':'Pittsburgh Steelers','SEA':'Seattle Seahawks',
 'SF':'San Francisco 49ers','TB':'Tampa Bay Buccaneers','TEN':'Tennessee Titans','WAS':'Washington Commanders'
}

def profit_per_dollar(odds):
    return odds/100 if odds>0 else 100/abs(odds) if odds<0 else float('nan')

@st.cache_data(ttl=300, show_spinner=False)
def fetch_odds(api_key):
    response = requests.get(API, params={'apiKey':api_key,'regions':'us','markets':'h2h,spreads,totals',
                                         'oddsFormat':'american','dateFormat':'iso'},timeout=25)
    response.raise_for_status()
    return response.json(), response.headers.get('x-requests-remaining','unknown')

def choose(rows,side,market,point=None):
    matches=[x for x in rows if x['market']==market and x['selection']==side and (point is None or x['point']==point)]
    return max(matches,key=lambda x:x['odds']) if matches else None

st.title('🏈 Live US Sportsbook Odds')
st.caption('NFL Model 5.1 companion • US bookmakers • Best available price per identical selection and line')
st.info('Research estimates only. Odds change, prices may be unavailable at your location, and model probabilities are not calibrated or proof of a profitable edge.')
try:
    key=st.secrets['ODDS_API_KEY']
    if not key or key=='YOUR_ACTUAL_API_KEY':raise KeyError('ODDS_API_KEY')
except (KeyError, FileNotFoundError):
    st.error('Set ODDS_API_KEY to your real key in Streamlit Cloud → App settings → Secrets.')
    st.stop()

if st.button('Refresh sportsbook odds'):
    fetch_odds.clear()
try:
    events, remaining=fetch_odds(key)
except requests.HTTPError as e:
    st.error(f'Odds provider rejected the request ({e.response.status_code}). Check your API key, subscription quota and market access.')
    st.stop()
except (requests.RequestException, ValueError) as e:
    st.error(f'Could not retrieve odds: {e}')
    st.stop()

st.caption(f'Live API request • Remaining API credits reported: {remaining} • Display refreshed at most every 5 minutes unless you click Refresh')
if not events:
    st.warning('The odds provider returned no NFL games. Try again closer to kickoff.')
    st.stop()

predictions=pd.DataFrame()
if PICKS.exists():
    try:predictions=pd.read_csv(PICKS)
    except (pd.errors.EmptyDataError, OSError):pass
if predictions.empty:
    st.warning('No saved Model 5.1 projections found. Run Predictions on the main page first. Market odds appear below without model edges.')

records=[]
for event in events:
    home,away=event.get('home_team'),event.get('away_team')
    if not home or not away:continue
    offerings=[]
    for book in event.get('bookmakers',[]):
        for market in book.get('markets',[]):
            mk=market.get('key')
            for outcome in market.get('outcomes',[]):
                price=outcome.get('price')
                if mk not in ('h2h','spreads','totals') or not isinstance(price,(int,float)) or price==0:continue
                offerings.append({'market':mk,'selection':outcome.get('name'),'point':outcome.get('point'),
                                  'odds':int(price),'book':book.get('title','Unknown')})
    p=None
    if not predictions.empty and {'home_team','away_team','home_win_probability'}.issubset(predictions):
        subset=predictions[(predictions.home_team.map(TEAMS)==home)&(predictions.away_team.map(TEAMS)==away)]
        if len(subset):p=subset.iloc[-1]
    home_ml=choose(offerings,home,'h2h');away_ml=choose(offerings,away,'h2h')
    spread_choices=[x for x in offerings if x['market']=='spreads' and x['selection']==home]
    total_choices=[x for x in offerings if x['market']=='totals' and x['selection']=='Over']
    # Spreads/totals: compare like-for-like lines, never label the highest payout
    # across different points as the universally best line.
    spread=max(spread_choices,key=lambda x:(x['point'] if x['point'] is not None else -999,x['odds'])) if spread_choices else None
    total=max(total_choices,key=lambda x:(-x['point'] if x['point'] is not None else -999,x['odds'])) if total_choices else None
    row={'Kickoff (UTC)':event.get('commence_time'),'Game':f'{away} @ {home}',
         'Home ML':home_ml['odds'] if home_ml else None,'Home ML book':home_ml['book'] if home_ml else None,
         'Away ML':away_ml['odds'] if away_ml else None,'Away ML book':away_ml['book'] if away_ml else None,
         'Home spread':spread['point'] if spread else None,'Spread odds':spread['odds'] if spread else None,
         'Spread book':spread['book'] if spread else None,
         'Total line':total['point'] if total else None,'Over odds':total['odds'] if total else None,
         'Total book':total['book'] if total else None}
    if p is not None:
        hp=float(p.home_win_probability)
        row['Model home win %']=round(hp*100,1)
        row['Model predicted total']=round(float(p.pred_total),1) if 'pred_total' in p and pd.notna(p.pred_total) else None
        row['Model home ML EV/$']=round(hp*profit_per_dollar(home_ml['odds'])-(1-hp),3) if home_ml else None
        row['Model away ML EV/$']=round((1-hp)*profit_per_dollar(away_ml['odds'])-hp,3) if away_ml else None
    records.append(row)

out=pd.DataFrame(records).sort_values('Kickoff (UTC)')
st.subheader('Best available NFL market prices')
st.dataframe(out,hide_index=True,use_container_width=True)
st.caption('Moneyline: highest American price for each team. Spread: most favorable home-team point line, then best price at that line. Total: lowest Over point line, then best price at that line. These are not necessarily the best Under or away-spread offers.')
st.download_button('Download live odds CSV',out.to_csv(index=False),'nfl_live_odds.csv','text/csv')
