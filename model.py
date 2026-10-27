"""NFL Model 4.0: pregame Elo and rolling form; benchmarked against 3.0 on identical test seasons.
No claims of live injury, weather, or sportsbook data without user-provided pregame CSVs.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, accuracy_score
import model3_baseline as baseline

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; ART=ROOT/'artifacts'
DATA.mkdir(exist_ok=True);ART.mkdir(exist_ok=True)
URL='https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
FEATURES=['home_winrate_5','away_winrate_5','home_pf_5','away_pf_5','home_pa_5','away_pa_5','home_margin_5','away_margin_5','home_total_5','away_total_5','home_rest','away_rest','home_games','away_games','neutral_site','wind_mph','temp_f','precip_mm','home_injury_points','away_injury_points', 'elo_home_diff', 'home_elo', 'away_elo', 'home_margin_10', 'away_margin_10', 'home_pf_10', 'away_pf_10', 'home_pa_10', 'away_pa_10', 'home_winrate_10', 'away_winrate_10', 'elo_home_prob']

def download():
    d=pd.read_csv(URL,low_memory=False)
    if not {'gameday','home_team','away_team','home_score','away_score'}.issubset(d):raise ValueError('Schedule source missing required columns')
    d.to_csv(DATA/'games.csv',index=False)
    print('Downloaded',len(d),'schedule rows')

def optional(name):
    path=DATA/name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def load():
    if not (DATA/'games.csv').exists():download()
    d=pd.read_csv(DATA/'games.csv',low_memory=False)
    d['gameday']=pd.to_datetime(d.gameday,errors='coerce')
    d=d.dropna(subset=['gameday','home_team','away_team']).copy()
    if 'game_type' in d:d=d[d.game_type=='REG'].copy()
    d['season']=pd.to_numeric(d.season,errors='coerce')
    d['week']=pd.to_numeric(d.week,errors='coerce')
    return d.sort_values(['gameday','game_id'] if 'game_id' in d else ['gameday','home_team']).reset_index(drop=True)

def build_features(games):
    weather=optional('weather.csv');injuries=optional('injuries.csv')
    for frame in (weather,injuries):
        if not frame.empty and 'game_id' in frame:frame['game_id']=frame.game_id.astype(str)
    history={};rows=[]; ratings={}; last_season={}
    # Elo is updated only after completed games; preseason ratings regress toward 1500.
    # Process an entire date before updating histories: same-day results cannot leak.
    for date,day in games.groupby('gameday',sort=True):
        updates=[]
        for _,g in day.iterrows():
            row=g.to_dict();season=int(g.season) if pd.notna(g.season) else -1
            for side in ('home','away'):
                team=g[f'{side}_team']
                if team not in ratings:ratings[team]=1500.
                if last_season.get(team)!=season:
                    ratings[team]=1500.+0.65*(ratings[team]-1500.)
                    last_season[team]=season
                row[f'{side}_elo']=ratings[team]
            home_adv=0 if str(g.get('location','Home')).lower()=='neutral' else 45
            row['elo_home_diff']=row['home_elo']+home_adv-row['away_elo']
            row['elo_home_prob']=1/(1+10**(-row['elo_home_diff']/400))
            for side in ('home','away'):
                team=g[f'{side}_team'];past=history.get(team,[])
                same=[x for x in past if x['season']==season]
                recent=same[-5:] or [x for x in past if x['season']==season-1][-5:]
                for key in ('winrate','pf','pa','margin','total'):
                    row[f'{side}_{key}_5']=float(np.mean([x[key] for x in recent])) if recent else np.nan
                for key in ('margin','pf','pa','winrate'):
                    last10=same[-10:] or [x for x in past if x['season']==season-1][-10:]
                    row[f'{side}_{key}_10']=float(np.mean([x[key] for x in last10])) if last10 else np.nan
                row[f'{side}_games']=len(same)
                row[f'{side}_rest']=(date-past[-1]['date']).days if past else np.nan
            row['neutral_site']=int(str(g.get('location','Home')).lower()=='neutral')
            for col,frame,fields in [('weather',weather,('wind_mph','temp_f','precip_mm')),('injuries',injuries,('home_injury_points','away_injury_points'))]:
                m=frame[frame.game_id==str(g.get('game_id',''))] if not frame.empty and 'game_id' in frame else pd.DataFrame()
                for field in fields:row[field]=pd.to_numeric(m.iloc[-1][field],errors='coerce') if len(m) and field in m else np.nan
            hs=pd.to_numeric(g.home_score,errors='coerce');aw=pd.to_numeric(g.away_score,errors='coerce')
            row['margin']=hs-aw if pd.notna(hs) and pd.notna(aw) else np.nan
            row['total']=hs+aw if pd.notna(hs) and pd.notna(aw) else np.nan
            row['home_win']=float(hs>aw) if pd.notna(hs) and pd.notna(aw) and hs!=aw else np.nan
            rows.append(row)
            if pd.notna(hs) and pd.notna(aw):
                # Store rating updates; no game on the same date sees another result.
                expected=row['elo_home_prob']
                actual=1. if hs>aw else (0. if hs<aw else .5)
                delta=22*(actual-expected)
                updates.append(('__elo__',g.home_team,g.away_team,delta))
                updates.extend([(g.home_team,dict(date=date,season=season,winrate=float(hs>aw),pf=hs,pa=aw,margin=hs-aw,total=hs+aw)),(g.away_team,dict(date=date,season=season,winrate=float(aw>hs),pf=aw,pa=hs,margin=aw-hs,total=hs+aw))])
        for item in updates:
            if len(item)==4 and item[0]=='__elo__':
                _,h,a,delta=item;ratings[h]+=delta;ratings[a]-=delta
            else:
                team,entry=item;history.setdefault(team,[]).append(entry)
    return pd.DataFrame(rows)

def fit(d,cutoff=None):
    train=d.dropna(subset=['margin','total','home_win']).copy()
    if cutoff is not None:train=train[train.season<cutoff]
    # Recent-season emphasis helps reduce drift from older NFL eras.
    max_year=int(train.season.max())
    train=train[train.season>=max_year-7].copy()
    if len(train)<350:raise ValueError('Need at least 350 historical completed games')
    x=train[FEATURES]
    models={
        'winner':make_pipeline(SimpleImputer(),StandardScaler(),LogisticRegression(max_iter=1500,C=.12)),
        'margin':make_pipeline(SimpleImputer(),HistGradientBoostingRegressor(max_iter=140,max_leaf_nodes=7,l2_regularization=30,random_state=21)),
        'total':make_pipeline(SimpleImputer(),HistGradientBoostingRegressor(max_iter=140,max_leaf_nodes=7,l2_regularization=30,random_state=22))}
    for name,target in [('winner','home_win'),('margin','margin'),('total','total')]:models[name].fit(x,train[target])
    return models

def backtest(d):
    """Evaluate both versions on the exact same completed seasons and games."""
    rows=[];preds=[];compare=[]
    years=[int(y) for y in sorted(d.season.dropna().unique()) if int(y)>=2022]
    base_d=baseline.build_features(baseline.load())
    for year in years:
        test=d[(d.season==year)&d.margin.notna()&d.home_win.notna()]
        base_test=base_d[(base_d.season==year)&base_d.margin.notna()&base_d.home_win.notna()]
        if len(test)<50 or len(d[(d.season<year)&d.margin.notna()])<350:continue
        models=fit(d,year);x=test[FEATURES]
        prob=models['winner'].predict_proba(x)[:,1];margin=models['margin'].predict(x);total=models['total'].predict(x)
        metrics=dict(season=year,games=len(test),winner_accuracy=accuracy_score(test.home_win,prob>=.5),winner_brier=brier_score_loss(test.home_win,prob),winner_log_loss=log_loss(test.home_win,np.clip(prob,.001,.999)),spread_mae=mean_absolute_error(test.margin,margin),total_mae=mean_absolute_error(test.total,total))
        rows.append(metrics);compare.append(dict(model='4.0',**metrics))
        for i,(_,g) in enumerate(test.iterrows()):preds.append(dict(season=year,game_id=g.get('game_id',''),actual_margin=g.margin,actual_total=g.total,actual_home_win=g.home_win,pred_margin=margin[i],pred_total=total[i],home_win_probability=prob[i]))
        if len(base_test)==len(test):
            bm=baseline.fit(base_d,year); bx=base_test[baseline.FEATURES]
            bp=bm['winner'].predict_proba(bx)[:,1];bmargin=bm['margin'].predict(bx);btotal=bm['total'].predict(bx)
            compare.append(dict(model='3.0',season=year,games=len(base_test),winner_accuracy=accuracy_score(base_test.home_win,bp>=.5),winner_brier=brier_score_loss(base_test.home_win,bp),winner_log_loss=log_loss(base_test.home_win,np.clip(bp,.001,.999)),spread_mae=mean_absolute_error(base_test.margin,bmargin),total_mae=mean_absolute_error(base_test.total,btotal)))
    cols=['season','games','winner_accuracy','winner_brier','winner_log_loss','spread_mae','total_mae']
    pd.DataFrame(rows,columns=cols).to_csv(ART/'backtest.csv',index=False)
    pd.DataFrame(preds).to_csv(ART/'backtest_predictions.csv',index=False)
    pd.DataFrame(compare,columns=['model']+cols).to_csv(ART/'model_comparison.csv',index=False)
    print('Compared Model 3.0 and 4.0 on',len(rows),'recent seasons')

def implied(odds):
    if pd.isna(odds) or odds==0:return np.nan
    return 100/(odds+100) if odds>0 else -odds/(100-odds)

def predict(d,models):
    upcoming=d[d.margin.isna()].copy()
    if upcoming.empty:
        print('No upcoming games available in schedule');return
    x=upcoming[FEATURES]
    upcoming['pred_margin_home']=models['margin'].predict(x)
    upcoming['pred_total']=models['total'].predict(x)
    upcoming['home_win_probability']=models['winner'].predict_proba(x)[:,1]
    upcoming['pred_home_score']=(upcoming.pred_total+upcoming.pred_margin_home)/2
    upcoming['pred_away_score']=(upcoming.pred_total-upcoming.pred_margin_home)/2
    odds=optional('odds.csv')
    if not odds.empty and 'game_id' in odds:
        fields=['book_total','over_odds','under_odds','home_ml','away_ml','book_home_spread']
        for field in fields:
            if field not in odds:odds[field]=np.nan
        upcoming=upcoming.merge(odds[['game_id']+fields].drop_duplicates('game_id',keep='last'),on='game_id',how='left')
    for field in ['book_total','home_ml','away_ml','book_home_spread']:
        if field not in upcoming:upcoming[field]=np.nan
        upcoming[field]=pd.to_numeric(upcoming[field],errors='coerce')
    upcoming['total_edge']=upcoming.pred_total-upcoming.book_total
    upcoming['total_pick']=np.select([upcoming.total_edge>=3,upcoming.total_edge<=-3],['OVER','UNDER'],default='PASS')
    upcoming.loc[upcoming.book_total.isna(),'total_pick']='NO LINE'
    upcoming['home_ml_implied']=upcoming.home_ml.map(implied)
    upcoming['away_ml_implied']=upcoming.away_ml.map(implied)
    upcoming['home_ml_probability_edge']=upcoming.home_win_probability-upcoming.home_ml_implied
    upcoming['away_ml_probability_edge']=1-upcoming.home_win_probability-upcoming.away_ml_implied
    upcoming['home_ml_ev_per_dollar']=upcoming.home_win_probability*upcoming.home_ml.map(lambda o: o/100 if pd.notna(o) and o>0 else (100/abs(o) if pd.notna(o) and o<0 else np.nan))-(1-upcoming.home_win_probability)
    upcoming['away_ml_ev_per_dollar']=(1-upcoming.home_win_probability)*upcoming.away_ml.map(lambda o: o/100 if pd.notna(o) and o>0 else (100/abs(o) if pd.notna(o) and o<0 else np.nan))-upcoming.home_win_probability
    upcoming['model_status']='V4_RESEARCH_UNCALIBRATED_NO_LIVE_ODDS'
    upcoming.to_csv(ART/'sunday_picks.csv',index=False)
    print('Wrote',len(upcoming),'future game projections')

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['download','train','backtest','predict','all']);a=p.parse_args()
    if a.command=='download':download();return
    if a.command=='all':download()  # refresh schedule when running full model
    d=build_features(load())
    if a.command in ('backtest','all'):backtest(d)
    if a.command in ('train','all'):
        models=fit(d);joblib.dump(models,ART/'models.joblib');print('Models trained')
    if a.command in ('predict','all'):
        path=ART/'models.joblib'
        if not path.exists():raise FileNotFoundError('Run model.py train first')
        predict(d,joblib.load(path))
if __name__=='__main__':main()
