import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, log_loss, brier_score_loss, accuracy_score

ROOT=Path(__file__).parent
DATA=ROOT/'data'; DATA.mkdir(exist_ok=True)
ART=ROOT/'artifacts'; ART.mkdir(exist_ok=True)
# NFLverse schedule contains scores and pregame metadata; never use closing lines as model inputs by default.
SCHEDULE='https://github.com/nflverse/nfldata/raw/refs/heads/master/data/games.csv'
# fallback to stable raw URL
SCHEDULE2='https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
FEATURES=['home_winrate_5','away_winrate_5','home_pf_5','away_pf_5','home_pa_5','away_pa_5','home_margin_5','away_margin_5','home_total_5','away_total_5','home_rest','away_rest','home_games','away_games','neutral_site','wind_mph','temp_f','precip_mm','home_injury_points','away_injury_points']

def download():
    for url in (SCHEDULE2,SCHEDULE):
        try:
            df=pd.read_csv(url,low_memory=False)
            if {'gameday','home_team','away_team','home_score','away_score'}.issubset(df.columns):
                df.to_csv(DATA/'games.csv',index=False);print('Downloaded',len(df),'games');return
        except Exception as e: print('Source unavailable:',str(e)[:180])
    raise RuntimeError('Download failed. Place nflverse games.csv at data/games.csv and retry.')

def load_games():
    p=DATA/'games.csv'
    if not p.exists(): download()
    d=pd.read_csv(p,low_memory=False)
    d['gameday']=pd.to_datetime(d['gameday'],errors='coerce')
    d=d.dropna(subset=['gameday','home_team','away_team']).copy()
    if 'game_type' in d: d=d[d.game_type.eq('REG')]
    d['season']=pd.to_numeric(d['season'],errors='coerce')
    d['week']=pd.to_numeric(d['week'],errors='coerce')
    return d.sort_values(['gameday','game_id'] if 'game_id' in d else ['gameday','home_team']).reset_index(drop=True)

def optional_csv(name):
    p=DATA/name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def extra_for_game(row,weather,injuries):
    key=str(row.get('game_id',''))
    vals={'wind_mph':np.nan,'temp_f':np.nan,'precip_mm':np.nan,'home_injury_points':0.,'away_injury_points':0.}
    for df,fields in [(weather,['wind_mph','temp_f','precip_mm']),(injuries,['home_injury_points','away_injury_points'])]:
        if not df.empty and 'game_id' in df:
            m=df[df.game_id.astype(str)==key]
            if len(m):
                for f in fields:
                    if f in m: vals[f]=pd.to_numeric(m.iloc[-1][f],errors='coerce')
    return vals

def features(d):
    weather=optional_csv('weather.csv');injuries=optional_csv('injuries.csv')
    # Only games before kickoff contribute to features. Season-to-season carryover is shrunk.
    history={};out=[]
    for _,g in d.iterrows():
        season=int(g.season) if pd.notna(g.season) else 0
        date=g.gameday
        row=g.to_dict()
        for side in ('home','away'):
            team=g[f'{side}_team']; prior=[v for v in history.get(team,[]) if v['date']<date]
            same=[v for v in prior if v['season']==season]
            recent=same[-5:]
            # Previous season stats are weak prior; avoid over-reliance on roster continuity.
            if not recent: recent=[dict(v,weight=.25) for v in prior[-5:] if v['season']==season-1]
            weights=np.array([v.get('weight',1.) for v in recent],float)
            for name,field in [('winrate_5','win'),('pf_5','pf'),('pa_5','pa'),('margin_5','margin'),('total_5','total')]:
                row[f'{side}_{name}']=float(np.average([v[field] for v in recent],weights=weights)) if len(recent) else np.nan
            row[f'{side}_games']=len(same)
            row[f'{side}_rest']=float((date-prior[-1]['date']).days) if prior else np.nan
        row['neutral_site']=int(str(g.get('location','Home')).lower()=='neutral')
        row.update(extra_for_game(g,weather,injuries))
        hs=pd.to_numeric(g.home_score,errors='coerce');aw=pd.to_numeric(g.away_score,errors='coerce')
        row['margin']=hs-aw if pd.notna(hs) and pd.notna(aw) else np.nan
        row['total']=hs+aw if pd.notna(hs) and pd.notna(aw) else np.nan
        row['home_win']=int(hs>aw) if pd.notna(hs) and pd.notna(aw) and hs!=aw else np.nan
        out.append(row)
        if pd.notna(hs) and pd.notna(aw):
            for side,team,pf,pa in [('home',g.home_team,hs,aw),('away',g.away_team,aw,hs)]:
                history.setdefault(team,[]).append(dict(date=date,season=season,win=float(pf>pa),pf=pf,pa=pa,margin=pf-pa,total=pf+pa))
    return pd.DataFrame(out)

def estimators():
    return {'margin':make_pipeline(SimpleImputer(add_indicator=False),HistGradientBoostingRegressor(max_iter=140,max_leaf_nodes=10,l2_regularization=12,random_state=42)),
            'total':make_pipeline(SimpleImputer(),HistGradientBoostingRegressor(max_iter=140,max_leaf_nodes=10,l2_regularization=12,random_state=43)),
            'winner':make_pipeline(SimpleImputer(),LogisticRegression(max_iter=1000,C=.3))}

def train(data,cutoff=None):
    complete=data.dropna(subset=['margin','total','home_win']).copy()
    if cutoff is not None: complete=complete[complete.season<cutoff]
    if len(complete)<200: raise ValueError('At least 200 completed historical games required.')
    models=estimators(); X=complete[FEATURES]
    for name,target in [('margin','margin'),('total','total'),('winner','home_win')]: models[name].fit(X,complete[target])
    # Residual dispersion estimated in-sample: approximate; use out-of-fold calibration for production betting.
    sigmas={name:float(np.std(complete[target]-models[name].predict(X),ddof=1)) for name,target in [('margin','margin'),('total','total')]}
    return models,sigmas

def backtest(d):
    years=sorted(int(y) for y in d.season.dropna().unique())
    rows=[]; predictions=[]
    for year in years:
        test=d[(d.season==year)&d.margin.notna()&d.home_win.notna()]
        train_games=d[(d.season<year)&d.margin.notna()]
        if len(train_games)<350 or len(test)<50:continue
        models,sigmas=train(d,year)
        X=test[FEATURES]; margin=models['margin'].predict(X);total=models['total'].predict(X);prob=models['winner'].predict_proba(X)[:,1]
        rows.append({'season':year,'games':len(test),'winner_accuracy':round(accuracy_score(test.home_win,prob>=.5),4),'winner_brier':round(brier_score_loss(test.home_win,prob),4),'winner_log_loss':round(log_loss(test.home_win,np.clip(prob,.001,.999)),4),'spread_mae':round(mean_absolute_error(test.margin,margin),3),'total_mae':round(mean_absolute_error(test.total,total),3)})
        for i,(_,g) in enumerate(test.iterrows()):predictions.append({'season':year,'game_id':g.get('game_id',''),'actual_margin':g.margin,'actual_total':g.total,'pred_margin':margin[i],'pred_total':total[i],'home_win_probability':prob[i]})
    pd.DataFrame(rows).to_csv(ART/'backtest.csv',index=False)
    pd.DataFrame(predictions).to_csv(ART/'backtest_predictions.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False) if rows else 'Insufficient seasons for walk-forward validation')

def predict(d,models,sigmas):
    upcoming=d[d.margin.isna()].copy()
    if upcoming.empty:print('No upcoming games in schedule');return
    X=upcoming[FEATURES]
    upcoming['pred_margin_home']=models['margin'].predict(X)
    upcoming['pred_total']=models['total'].predict(X)
    upcoming['home_win_probability']=models['winner'].predict_proba(X)[:,1]
    upcoming['pred_home_score']=(upcoming.pred_total+upcoming.pred_margin_home)/2
    upcoming['pred_away_score']=(upcoming.pred_total-upcoming.pred_margin_home)/2
    # Odds and injury/weather feeds are optional; no invented odds or claimed betting edge.
    odds=optional_csv('odds.csv')
    if not odds.empty and 'game_id' in odds:
        for f in ['book_total','over_odds','under_odds','home_ml','away_ml','book_home_spread']:
            if f not in odds:odds[f]=np.nan
        upcoming=upcoming.merge(odds[['game_id','book_total','over_odds','under_odds','home_ml','away_ml','book_home_spread']].drop_duplicates('game_id',keep='last'),on='game_id',how='left')
        upcoming['total_edge']=upcoming.pred_total-upcoming.book_total
        from math import erf,sqrt
        # Normal approximation; sigma is in-sample and not a calibrated probability.
        upcoming['over_probability_est']=upcoming.apply(lambda r: .5*(1+erf((r.pred_total-r.book_total)/(max(sigmas['total'],1)*sqrt(2)))) if pd.notna(r.book_total) else np.nan,axis=1)
        upcoming['under_probability_est']=1-upcoming.over_probability_est
        upcoming['total_pick']=np.where(upcoming.total_edge>=3,'OVER',np.where(upcoming.total_edge<=-3,'UNDER','PASS'))
    else:
        upcoming['total_edge']=np.nan;upcoming['total_pick']='NO LINE'
    upcoming['model_status']='TRAINED_BASELINE_NOT_LIVE_VALIDATED'
    upcoming.to_csv(ART/'sunday_picks.csv',index=False)
    print('Wrote',len(upcoming),'future games to',ART/'sunday_picks.csv')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['download','train','backtest','predict','all']);a=ap.parse_args()
    if a.command=='download':download();return
    if a.command=='all' and not (DATA/'games.csv').exists():download()
    d=features(load_games())
    if a.command in ('backtest','all'):backtest(d)
    if a.command in ('train','all'):
        models,sigmas=train(d);joblib.dump({'models':models,'sigmas':sigmas,'features':FEATURES},ART/'models.joblib');print('Saved trained models')
    if a.command in ('predict','all'):
        p=ART/'models.joblib'
        if not p.exists():raise FileNotFoundError('Run python model.py train first')
        pack=joblib.load(p);predict(d,pack['models'],pack['sigmas'])
if __name__=='__main__':main()
