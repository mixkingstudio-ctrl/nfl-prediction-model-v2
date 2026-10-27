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
                api_key=st.secrets.get('ODDS_API_KEY','')
                if api_key and api_key!='YOUR_ACTUAL_API_KEY':
                    p,live_matches=attach_live_odds(p,get_live_moneylines(api_key))
                    if live_matches:
                        st.success(f'Live sportsbook moneylines matched to {live_matches} game(s).')
                    else:
                        st.info('Live odds loaded, but no matching games were found for this selected week.')
                else:
                    st.info('No ODDS_API_KEY configured. Moneyline EV will remain unavailable.')
            except (requests.RequestException,ValueError,KeyError) as exc:
                st.warning(f'Live odds unavailable right now: {exc}')
            except Exception as exc:
                st.warning(f'Could not match sportsbook odds to predictions: {exc}')
            p['favorite']=p.apply(lambda r:r.home_team if r.home_win_probability>=.5 else r.away_team,axis=1)
            p['favorite_probability']=p.home_win_probability.where(p.home_win_probability>=.5,1-p.home_win_probability)
            st.subheader('💵 Moneyline')
            ml=['gameday','away_team','home_team','favorite','favorite_probability','home_ml','away_ml','home_ml_book','away_ml_book','home_ml_probability_edge','away_ml_probability_edge','home_ml_ev_per_dollar','away_ml_ev_per_dollar']
            st.dataframe(p[[c for c in ml if c in p]],hide_index=True,use_container_width=True)
            st.caption('Model 4.0 winner engine restored. Live US moneylines are matched when available. Model probabilities are uncalibrated; estimated EV is not proof of a profitable edge.')
            st.subheader('🎯 Top three market opportunities')
            if {'home_ml_ev_per_dollar','away_ml_ev_per_dollar'}.issubset(p):
                opportunities=[]
                for _,g in p.iterrows():
                    for side in ('home','away'):
                        ev=g.get(f'{side}_ml_ev_per_dollar')
                        odd=g.get(f'{side}_ml')
                        if pd.notna(ev) and pd.notna(odd) and ev>0:
                            opportunities.append({'gameday':g.gameday,'game':f'{g.away_team} @ {g.home_team}',
                                'selection':g[f'{side}_team'],'american_odds':odd,'estimated_ev_per_dollar':ev})
                if opportunities:
                    st.dataframe(pd.DataFrame(opportunities).sort_values('estimated_ev_per_dollar',ascending=False).head(3),hide_index=True,use_container_width=True)
                    st.caption('Research estimates only; not validated profitable bets. Check sportsbook prices and timing.')
                else:st.info('No positive-EV opportunities with supplied sportsbook moneyline odds. Missing odds are never treated as betting edges.')
            st.subheader('🏆 Top three predicted winners')
            st.dataframe(p.sort_values('favorite_probability',ascending=False)[['gameday','away_team','home_team','favorite','favorite_probability','pred_away_score','pred_home_score']].head(3),hide_index=True,use_container_width=True)
            st.subheader('📊 Over/Under')
            cols=['gameday','away_team','home_team','pred_total','book_total','total_pick','total_edge','over_probability_est','under_probability_est']
            st.dataframe(p[[c for c in cols if c in p]],hide_index=True,use_container_width=True)
            st.subheader('📈 Spreads')
            cols=['gameday','away_team','home_team','pred_margin_home','book_home_spread','pred_home_score','pred_away_score']
            st.dataframe(p[[c for c in cols if c in p]],hide_index=True,use_container_width=True)
            st.caption('Predicted home margin = home score minus away score. A negative number favors the away team.')
            st.download_button('Download all picks CSV',p.to_csv(index=False),'sunday_picks.csv')
    st.warning('Research release: Moneyline restored from 4.0; totals ensemble is experimental. Not validated betting advice. EPA, live injuries and weather are NOT automatically connected. Live moneyline odds are fetched when an API key is configured. Optional CSV inputs must be sourced pregame.')
with t2:
    bt=read(ART/'backtest.csv')
    if bt.empty:st.info('Run Predictions to generate chronological season-by-season test results.')
    else:
        st.dataframe(bt,hide_index=True,use_container_width=True)
        if {'season','spread_mae','total_mae'}.issubset(bt):st.line_chart(bt.set_index(bt['season'].astype(str))[['spread_mae','total_mae']])
        st.caption('Walk-forward out-of-sample tests. Accuracy and Brier score do not establish profitable betting.')
        comparison=read(ART/'model_comparison.csv')
        if not comparison.empty:
            st.subheader('4.0 versus 5.1 — same test seasons')
            st.dataframe(comparison.sort_values(['season','model']),hide_index=True,use_container_width=True)
            st.caption('Winner accuracy: higher is better. Brier, spread MAE and total MAE: lower is better. Compare on the same seasons.')
with t3:
    path=DATA/'bets.csv';bets=read(path)
    with st.form('bet'):
        a,b,c=st.columns(3)
        with a:game=st.text_input('Game');selection=st.text_input('Selection')
        with b:odds=st.number_input('American odds',value=-110,step=5);stake=st.number_input('Stake ($)',min_value=0.,value=20.,step=5.)
        with c:result=st.selectbox('Result',['PENDING','WIN','LOSS','PUSH']);date=st.date_input('Date')
        save=st.form_submit_button('Add bet')
    if save:
        if not game or not selection:st.error('Enter game and selection.')
        else:
            entry=pd.DataFrame([dict(date=str(date),game=game,selection=selection,american_odds=odds,stake=stake,result=result)])
            bets=pd.concat([bets,entry],ignore_index=True);bets.to_csv(path,index=False)
            st.success('Bet saved temporarily. Download the log regularly: Streamlit cloud storage is ephemeral.')
    if not bets.empty:
        def profit(r):
            if r.result=='WIN':return r.stake*(r.american_odds/100 if r.american_odds>0 else 100/abs(r.american_odds))
            if r.result=='LOSS':return -r.stake
            return 0.
        bets['profit']=bets.apply(profit,axis=1)
        settled=bets[bets.result.isin(['WIN','LOSS','PUSH'])];risk=settled.stake.sum()
        x,y,z=st.columns(3)
        x.metric('Net profit',f'${settled.profit.sum():.2f}')
        y.metric('ROI',f'{100*settled.profit.sum()/risk:.1f}%' if risk else 'N/A')
        z.metric('Settled bets',len(settled))
        st.dataframe(bets,hide_index=True,use_container_width=True)
        st.download_button('Export bet log',bets.to_csv(index=False),'bet_log.csv')
