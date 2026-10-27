from pathlib import Path
import pandas as pd
import streamlit as st
ROOT=Path(__file__).parent; ART=ROOT/'artifacts'; DATA=ROOT/'data'; DATA.mkdir(exist_ok=True)
st.set_page_config(page_title='NFL Weekly Report Dashboard 2.0',layout='wide')
st.title('🏈 NFL Weekly Report Dashboard 2.0')
st.caption('Moneyline • Over/Under • Walk-forward validation • Bet tracking')
def read(path):
    try:return pd.read_csv(path)
    except (FileNotFoundError,pd.errors.EmptyDataError):return pd.DataFrame()

picks=read(ART/'sunday_picks.csv');bt=read(ART/'backtest.csv'); betsfile=DATA/'bets.csv';bets=read(betsfile)
t1,t2,t3=st.tabs(['Sunday predictions','Backtesting','Bet tracker'])
with t1:
    if picks.empty:st.info('Run python model.py all to create predictions.')
    else:
        seasons=sorted(picks.season.dropna().unique().tolist())
        sel=st.selectbox('Season',seasons,index=len(seasons)-1) if seasons else None
        p=picks[picks.season==sel] if sel is not None else picks
        if 'week' in p:
            weeks=sorted(p.week.dropna().unique().tolist());week=st.selectbox('Week',weeks,index=0) if weeks else None
            if week is not None:p=p[p.week==week]
        p=p.copy();p['favorite']=p.apply(lambda r:r.home_team if r.home_win_probability>=.5 else r.away_team,axis=1)
        p['favorite_probability']=p.home_win_probability.where(p.home_win_probability>=.5,1-p.home_win_probability)
        st.subheader('Top three predicted winners')
        st.dataframe(p.sort_values('favorite_probability',ascending=False)[['gameday','away_team','home_team','favorite','favorite_probability','pred_away_score','pred_home_score']].head(3),hide_index=True,use_container_width=True)
        st.subheader('Totals: model edge vs market')
        cols=['gameday','away_team','home_team','pred_total','total_pick','total_edge']
        if 'book_total' in p:cols.insert(4,'book_total')
        st.dataframe(p.sort_values('total_edge',key=lambda x:x.abs(),ascending=False)[cols],hide_index=True,use_container_width=True)
        st.download_button('Download all picks (CSV)',p.to_csv(index=False),'sunday_picks.csv')
        st.warning('Baseline predictions are not calibrated betting advice. Missing odds/weather/injuries are not silently treated as live feeds.')
with t2:
    if bt.empty:st.info('Run python model.py backtest first.')
    else:
        st.dataframe(bt,use_container_width=True,hide_index=True)
        st.line_chart(bt.set_index('season')[['spread_mae','total_mae']])
        st.caption('Chronological season-by-season out-of-sample testing; no random train/test split.')
with t3:
    with st.form('bet'):
        a,b,c=st.columns(3)
        with a:game=st.text_input('Game');selection=st.text_input('Selection (e.g. UNDER 44.5)')
        with b:odds=st.number_input('American odds',value=-110,step=5);stake=st.number_input('Stake ($)',min_value=0.,value=20.,step=5.)
        with c:result=st.selectbox('Result',['PENDING','WIN','LOSS','PUSH']);date=st.date_input('Date')
        save=st.form_submit_button('Add bet')
    if save:
        if not game or not selection:st.error('Enter game and selection.')
        else:
            entry=pd.DataFrame([{'date':str(date),'game':game,'selection':selection,'american_odds':odds,'stake':stake,'result':result}]);bets=pd.concat([bets,entry],ignore_index=True);bets.to_csv(betsfile,index=False);st.success('Bet recorded')
    if not bets.empty:
        def profit(r):
            if r.result=='WIN':return r.stake*(r.american_odds/100 if r.american_odds>0 else 100/abs(r.american_odds))
            if r.result=='LOSS':return -r.stake
            return 0.
        bets['profit']=bets.apply(profit,axis=1)
        settled=bets[bets.result.isin(['WIN','LOSS','PUSH'])];risk=settled.stake.sum()
        x,y,z=st.columns(3);x.metric('Net profit',f"${settled.profit.sum():.2f}");y.metric('ROI',f"{100*settled.profit.sum()/risk:.1f}%" if risk else 'N/A');z.metric('Settled bets',len(settled))
        st.dataframe(bets,use_container_width=True,hide_index=True)
        st.download_button('Export bet log',bets.to_csv(index=False),'bet_log.csv')
