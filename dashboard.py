from pathlib import Path
import subprocess,sys
import pandas as pd
import requests
import streamlit as st
ROOT=Path(__file__).resolve().parent
ART=ROOT/'artifacts';DATA=ROOT/'data';DATA.mkdir(exist_ok=True)
st.set_page_config(page_title='NFL Prediction Model 5.1 (Research)',layout='wide')
st.title('🏈 NFL Prediction Model 5.1 (Research)')
st.caption('Moneyline • Over/Under • Spreads • Walk-forward testing • Bet tracker')
def read(path):
    try:return pd.read_csv(path)
    except (FileNotFoundError,pd.errors.EmptyDataError):return pd.DataFrame()
# Live US sportsbook moneyline comparison; never modifies model predictions.
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
@st.cache_data(ttl=300,show_spinner=False)
def get_live_moneylines(key):
    url='https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/'
    response=requests.get(url,params={'apiKey':key,'regions':'us','markets':'h2h',
        'oddsFormat':'american','dateFormat':'iso'},timeout=25)
    response.raise_for_status()
    return response.json()

def moneyline_profit(odds):
    return odds/100 if odds>0 else 100/abs(odds)

def attach_live_odds(frame,events):
    frame=frame.copy()
    for field in ('home_ml','away_ml','home_ml_book','away_ml_book','home_ml_probability_edge','away_ml_probability_edge',
                  'home_ml_ev_per_dollar','away_ml_ev_per_dollar'):
        frame[field]=float('nan')
    frame['home_ml_book']=''
    frame['away_ml_book']=''
    by_match={}
    for event in events:
        if not event.get('home_team') or not event.get('away_team'):continue
        offers={}
        for bookmaker in event.get('bookmakers',[]):
            for market in bookmaker.get('markets',[]):
                if market.get('key')!='h2h':continue
                for outcome in market.get('outcomes',[]):
                    price=outcome.get('price')
                    name=outcome.get('name')
                    if isinstance(price,(int,float)) and price!=0 and name:
                        if name not in offers or price>offers[name][0]:
                            offers[name]=(int(price),bookmaker.get('title',''))
        by_match[(event['away_team'],event['home_team'])]=offers
    matched=0
    for idx,game in frame.iterrows():
        offers=by_match.get((TEAMS.get(game['away_team']),TEAMS.get(game['home_team'])))
        if not offers:continue
        hp=pd.to_numeric(game.get('home_win_probability'),errors='coerce')
        if pd.isna(hp) or not 0<=hp<=1:continue
        matched+=1
        for side,prob in (('home',hp),('away',1-hp)):
            team=TEAMS.get(game[f'{side}_team'])
            if team not in offers:continue
            odds,book=offers[team]
            frame.at[idx,f'{side}_ml']=odds
            frame.at[idx,f'{side}_ml_book']=book
            implied=100/(odds+100) if odds>0 else abs(odds)/(abs(odds)+100)
            frame.at[idx,f'{side}_ml_probability_edge']=prob-implied
            frame.at[idx,f'{side}_ml_ev_per_dollar']=prob*moneyline_profit(odds)-(1-prob)
    return frame,matched

t1,t2,t3=st.tabs(['Sunday picks','Backtesting','Bet tracker'])
with t1:
    if st.button('🏈 Run Predictions',type='primary'):
        with st.spinner('Refreshing schedule, training models and testing historical seasons...'):
            try:
                result=subprocess.run([sys.executable,str(ROOT/'model.py'),'all'],cwd=ROOT,capture_output=True,text=True,timeout=900)
                if result.returncode:st.error('Model run failed');st.code((result.stderr or result.stdout)[-6000:])
                else:st.success('Predictions generated');st.rerun()
            except Exception as exc:st.error(f'Model run could not complete: {exc}')
    picks=read(ART/'sunday_picks.csv')
    if picks.empty:st.info('No predictions yet. Click Run Predictions. The schedule source must include future games.')
    else:
        seasons=sorted(picks.season.dropna().unique().tolist())
        season=st.selectbox('Season',seasons,index=len(seasons)-1) if seasons else None
        p=picks[picks.season==season].copy() if season is not None else picks.copy()
        if 'week' in p and len(p):
            weeks=sorted(p.week.dropna().unique().tolist())
            if weeks:p=p[p.week==st.selectbox('Week',weeks,index=0)]
        if p.empty:st.info('No games for this selection.')
        else:
            live_matches=0
            try:
