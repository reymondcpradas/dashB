import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title='Agusan del Sur | Palay Analytics', page_icon='🌾', layout='wide')
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800&display=swap');
html,body,[class*="css"],.stApp{font-family:'DM Sans',sans-serif!important}
.stApp,[data-testid="stAppViewContainer"]{background:linear-gradient(145deg,#080d17,#101a2a 55%,#0b1422)!important;color:#edf4ff!important}
.block-container{padding-top:1.4rem;padding-bottom:3rem;max-width:1450px}
h1,h2,h3,h4{font-family:'Outfit',sans-serif!important;letter-spacing:-.02em;color:#f5f9ff!important}
p,li,label,[data-testid="stMarkdownContainer"],.stCaption{color:#f1f5fc!important}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#101d30,#13263a 80%,#101b2c)!important;border-right:1px solid #31435b!important}
[data-testid="stSidebar"] *{color:#f0f6ff!important}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){background:#234a50;border:1px solid #54d3b0;border-radius:10px}
[data-testid="stMetric"]{background:#19283b;border:1px solid #344a60;border-radius:17px;padding:18px 19px;box-shadow:0 8px 24px #050b1666}
[data-testid="stMetricLabel"] *{color:#e1eaf7!important;font-weight:700!important}
[data-testid="stMetricValue"] *{color:#8ff0ce!important;font-family:'Outfit',sans-serif;font-weight:800!important}
[data-testid="stPlotlyChart"],[data-testid="stDataFrame"]{background:#172538;border:1px solid #344a60;border-radius:16px;padding:9px}
[data-testid="stAlert"]{border-radius:12px}
.stDownloadButton button,.stButton button{background:#126e61!important;color:#fff!important;border:1px solid #4acbb2!important;border-radius:10px!important;font-weight:700!important}
.stDownloadButton button:hover,.stButton button:hover{background:#1c927e!important}
.hero{background:radial-gradient(circle at 88% 15%,#2a78604d,transparent 37%),linear-gradient(120deg,#173a42,#144b48 65%,#195765);border:1px solid #397b7a;border-radius:22px;padding:27px 32px;color:#fff;margin-bottom:18px;box-shadow:0 12px 32px #050b1655}
.hero .eyebrow{font-size:.76rem;letter-spacing:.17em;font-weight:800;color:#a7f4ce;margin-bottom:9px}
.hero .heading{font-family:'Outfit',sans-serif;font-size:2rem;font-weight:800;line-height:1.17;color:#fff;letter-spacing:-.03em}
.hero .sub{color:#e0f0ed;font-size:.95rem;margin-top:10px}
.pill{display:inline-block;border:1px solid #9ee9d366;border-radius:30px;padding:5px 12px;margin-top:16px;margin-right:8px;font-size:.76rem;color:#f0ffff;background:#ffffff12}
.section-intro{color:#cbd9e8;font-size:.94rem;margin-bottom:15px}
[data-baseweb="select"]>div,[data-baseweb="input"]>div,textarea{background:#1b2c40!important;color:#f5faff!important;border-color:#52677f!important}
[data-testid="stFileUploader"], [data-testid="stExpander"]{background:#172538!important;border:1px solid #344a60!important;border-radius:12px}
@media(max-width:640px){.hero{padding:22px}.hero .heading{font-size:1.45rem}}

/* Accessible dark-mode inputs and tables */
[data-testid="stDataFrame"]{color:#f5f8ff!important}
[data-testid="stDataFrame"] *{color:inherit}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"]{font-weight:700}
[data-baseweb="select"] input,[data-baseweb="input"] input{color:#ffffff!important;-webkit-text-fill-color:#ffffff!important}
[data-baseweb="select"] svg{fill:#e7f4ff!important}
[data-testid="stRadio"] label{padding:7px 8px;border-radius:9px}
[data-testid="stRadio"] label:hover{background:#263b53}
[data-testid="stTabs"] button{color:#f1f5ff!important}
[data-testid="stAlert"] p{color:#f6faff!important}
</style>""",unsafe_allow_html=True)

COLS=['Production','Harvested Area','Yield','Farmgate Price']
UNITS={'Production':'MT','Harvested Area':'ha','Yield':'MT/ha','Farmgate Price':'₱/kg','Rainfall':'mm','Temperature':'°C'}
annual=pd.DataFrame({'Year':[2021,2022,2023,2024,2025], 'Production':[270963.35,232758,288638,306775,326394], 'Harvested Area':[83634,80358,82211,87039,83555], 'Yield':[3.24,2.90,3.51,3.52,3.91], 'Farmgate Price':[14.96,15.28,18.73,19.76,14.9325], 'Rainfall':[3729,4494.4,3615.9,2919.1,3553.6], 'Temperature':[26.15,25.80,26.25,26.65,25.85]})
sem=pd.DataFrame([[2021,1,1877,26.0,'Wet Climate'],[2021,2,1852,26.3,'Dry Climate'],[2022,1,2192,25.6,'Wet Climate'],[2022,2,2302.4,26.0,'Wet Climate'],[2023,1,1913.9,26.0,'Wet Climate'],[2023,2,1702,26.5,'Dry Climate'],[2024,1,1081.5,26.8,'Dry Climate'],[2024,2,1837.6,26.5,'Dry Climate'],[2025,1,1791.6,25.6,'Dry Climate'],[2025,2,1762,26.1,'Dry Climate']],columns=['Year','Semester','Rainfall','Temperature','Classification'])
forecast=pd.DataFrame({'Year':[2026,2027,2028,2029,2030], 'Production':[340569.16,359056.99,377544.82,396032.65,414520.48], 'Harvested Area':[85316.30,85968.60,86620.90,87273.20,87925.50], 'Yield':[4.00,4.20,4.40,4.59,4.79], 'Farmgate Price':[18.06,18.50,18.95,19.39,19.83]})

reported_growth=pd.DataFrame({'Year':[2021,2022,2023,2024,2025],'Production':[np.nan,-14.10,24.01,6.28,6.40],'Harvested Area':[np.nan,-3.92,2.31,5.87,-4.00],'Yield':[np.nan,-10.60,21.21,.39,10.83],'Farmgate Price':[np.nan,2.07,22.62,5.53,-24.43]})
reported_corr=pd.DataFrame([[1,.676,.980,.298,-.811,.348],[.676,1,.517,.543,-.942,.808],[.980,.517,1,.201,-.691,.191],[.298,.543,.201,1,-.664,.864],[-.811,-.942,-.691,-.664,1,-.822],[.348,.808,.191,.864,-.822,1]],index=COLS+['Rainfall','Temperature'],columns=COLS+['Rainfall','Temperature'])

def csv_button(label, df, name):
    st.download_button(label,df.to_csv(index=True).encode('utf-8-sig'),file_name=name,mime='text/csv')
def historical_chart(df, variable):
    colors = {'Production':'#5be0b1','Harvested Area':'#60a5fa',
              'Yield':'#ffc46b','Farmgate Price':'#ff8589'}
    units = {'Production':'metric tons','Harvested Area':'hectares',
             'Yield':'mt/ha','Farmgate Price':'₱/kg'}
    color = colors[variable]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Year'], y=df[variable], mode='lines+markers',
        line=dict(color=color, width=4), marker=dict(color=color, size=10),
        showlegend=False,
        hovertemplate='%{x}: %{y:,.4f}<extra></extra>'))
    fig.update_layout(
        template='plotly_dark', height=480, autosize=True,
        paper_bgcolor='#102231', plot_bgcolor='#102231',
        font=dict(color='#f4f8ff',size=15,family='DM Sans'),
        margin=dict(l=30,r=20,t=38,b=38),
        xaxis=dict(title='Year',tickmode='array',tickvals=list(range(2021,2026)),
                   range=[2020.65,2025.35],showgrid=False,zeroline=False,
                   title_font=dict(size=17),tickfont=dict(size=14)),
        yaxis=dict(title=units[variable],gridcolor='#30495a',zeroline=False,
                   title_font=dict(size=17),tickfont=dict(size=14)),
        hovermode='x unified',hoverlabel=dict(bgcolor='#1c3448',font_color='#ffffff'))
    return fig


def forecasting_chart(df, variable, future):
    colors = {'Production':'#5be0b1','Harvested Area':'#60a5fa',
              'Yield':'#ffc46b','Farmgate Price':'#ff8589'}
    unit = {'Production':'metric tons','Harvested Area':'hectares',
            'Yield':'mt/ha','Farmgate Price':'₱/kg'}[variable]
    model = LinearRegression().fit(df[['Year']], df[variable])
    fitted = model.predict(df[['Year']])
    color = colors[variable]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Year'], y=df[variable], mode='lines+markers',
        name='Observed 2021–2025', line=dict(color=color,width=4),
        marker=dict(size=10,color=color),hovertemplate='%{x}: %{y:,.4f}<extra>Observed</extra>'))
    fig.add_trace(go.Scatter(x=df['Year'],y=fitted,mode='lines',
        name='Historical fitted trend',line=dict(color='#a0acbb',width=2.5,dash='dot'),
        hovertemplate='%{x}: %{y:,.4f}<extra>Fitted</extra>'))
    fig.add_trace(go.Scatter(x=future['Year'],y=future[variable],mode='lines+markers',
        name='Projected 2026–2030',line=dict(color='#ffc46b',width=4,dash='dash'),
        marker=dict(size=10,color='#ffc46b'),
        hovertemplate='%{x}: %{y:,.4f}<extra>Projected</extra>'))
    fig.add_vline(x=2025.5,line_color='#8393a5',line_width=2,line_dash='dot')
    fig.update_layout(
        template='plotly_dark',height=490,autosize=True,
        paper_bgcolor='#102231',plot_bgcolor='#102231',
        font=dict(color='#f4f8ff',size=15,family='DM Sans'),
        margin=dict(l=30,r=20,t=72,b=38),
        legend=dict(orientation='h',y=1.15,x=0,xanchor='left',
                    font=dict(size=14,color='#ffffff'),bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(title='Year',tickmode='array',tickvals=list(range(2021,2031)),
                   range=[2020.4,2030.6],showgrid=False,zeroline=False,
                   title_font=dict(size=17),tickfont=dict(size=14)),
        yaxis=dict(title=unit,gridcolor='#30495a',zeroline=False,
                   title_font=dict(size=17),tickfont=dict(size=14)),
        hovermode='x unified',hoverlabel=dict(bgcolor='#1c3448',font_color='#ffffff'))
    return fig

def calculate_metrics(df):
    rows=[]
    for col in COLS:
        model=LinearRegression().fit(df[['Year']],df[col]); fitted=model.predict(df[['Year']])
        rows.append([col,r2_score(df[col],fitted),mean_absolute_error(df[col],fitted),np.sqrt(mean_squared_error(df[col],fitted)),float(model.coef_[0])])
    return pd.DataFrame(rows,columns=['Indicator','R²','MAE','RMSE','Slope/year'])

st.sidebar.title('🌾 PALAY INSIGHTS')
st.sidebar.caption('RESEARCH ANALYTICS  •  2021–2030')
source=st.sidebar.radio('Data mode',['Manuscript reference','Upload and recalculate'])
df=annual.copy()
if source=='Upload and recalculate':
    uploaded=st.sidebar.file_uploader('Upload annual dataset (CSV or XLSX)',type=['csv','xlsx'])
    if uploaded:
        try:
            raw=pd.read_csv(uploaded) if uploaded.name.lower().endswith('.csv') else pd.read_excel(uploaded)
            st.sidebar.caption('Required columns: Year, Production, Harvested Area, Yield, Farmgate Price. Optional: Rainfall, Temperature.')
            raw.columns=[str(c).strip() for c in raw.columns]
            missing=[c for c in ['Year']+COLS if c not in raw.columns]
            if missing: st.sidebar.error('Missing columns: '+', '.join(missing))
            else:
                for c in ['Year']+COLS+['Rainfall','Temperature']:
                    if c in raw: raw[c]=pd.to_numeric(raw[c].astype(str).str.replace(',','',regex=False),errors='coerce')
                before=len(raw); raw=raw.drop_duplicates().dropna(subset=['Year']+COLS)
                raw=raw[(raw.Year>=2021)&(raw.Year<=2025)].sort_values('Year')
                if raw.Year.duplicated().any(): st.sidebar.error('Duplicate years detected: provide one annual row per year.')
                elif len(raw)!=5 or set(raw.Year.astype(int))!=set(range(2021,2026)): st.sidebar.error('Please provide exactly one valid annual row for each year 2021–2025.')
                else:
                    df=raw.reset_index(drop=True)
                    st.sidebar.success(f'Validated 5 annual rows; {before-len(raw)} invalid/duplicate rows removed.')
        except Exception as e: st.sidebar.error(f'Could not load file: {e}')
    else: st.sidebar.info('Upload a dataset to replace the built-in reference values.')

is_reference=source=='Manuscript reference' or df.equals(annual)
if source=='Upload and recalculate' and not df.equals(annual):
    frows=[]
    for year in range(2026,2031):
        frows.append({'Year':year,**{c:float(LinearRegression().fit(df[['Year']],df[c]).predict(pd.DataFrame({'Year':[year]}))[0]) for c in COLS}})
    active_forecast=pd.DataFrame(frows)
    active_metrics=calculate_metrics(df)
else:
    active_forecast=forecast.copy();active_metrics=calculate_metrics(df)

pages=['Overview','Data Preparation & Statistics','Historical Trends','Growth Rates','Correlation & Climate','Forecasting','Model Evaluation','Decision Support']
page=st.sidebar.radio('Navigation',pages)
st.sidebar.divider()

st.markdown('''<div class="hero"><div class="eyebrow">AGUSAN DEL SUR • AGRICULTURAL INTELLIGENCE</div><div class="heading">🌾 Palay Production Analytics<br>and Forecasting Dashboard</div><div class="sub">Chapter 4 Results · Historical Trends 2021–2025 · Forecasts 2026–2030</div><span class="pill">📈 4 production indicators</span><span class="pill">🔎 Research analytics</span><span class="pill">📊 Data-driven insights</span></div>''',unsafe_allow_html=True)

if page=='Overview':
    st.subheader('2025 Agricultural Indicators')
    latest=df.iloc[-1]; prev=df.iloc[-2]
    cols=st.columns(4)
    for widget,col in zip(cols,COLS):
        with widget: st.metric(col,f'{latest[col]:,.4f}' if col=='Farmgate Price' else f'{latest[col]:,.2f}',f'{(latest[col]/prev[col]-1)*100:+.2f}% vs 2024')
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(historical_chart(df,'Production'),use_container_width=True)
    with c2: st.plotly_chart(historical_chart(df,'Farmgate Price'),use_container_width=True)
    st.subheader('Annual Data')
    st.dataframe(df.style.format(precision=2),use_container_width=True)

elif page=='Data Preparation & Statistics':
    st.header('Data Preparation and Descriptive Statistics')
    st.dataframe(df,use_container_width=True,hide_index=True)
    st.subheader('Quality Checks')
    st.write(pd.DataFrame({'Check':['Annual records','Missing values','Duplicate years','Years covered','Nonpositive production/area/yield'], 'Result':[len(df),int(df.isna().sum().sum()),int(df.Year.duplicated().sum()),f'{int(df.Year.min())}–{int(df.Year.max())}',int((df[['Production','Harvested Area','Yield']]<=0).sum().sum())]}))
    st.subheader('Descriptive Statistics')
    stats=df[COLS+[c for c in ['Rainfall','Temperature'] if c in df]].agg(['mean','std','min','max']).T
    stats.columns=['Mean','Standard Deviation (sample)','Minimum','Maximum']
    st.dataframe(stats.style.format('{:,.3f}'),use_container_width=True)
    csv_button('Download annual dataset',df,'annual_palay_data.csv')
    csv_button('Download descriptive statistics',stats,'descriptive_statistics.csv')

elif page=='Historical Trends':
    st.header('Historical Trend Analysis (2021–2025)')
    for variable in COLS:
        with st.expander(variable, expanded=True):
            st.plotly_chart(historical_chart(df,variable),use_container_width=True)
    st.dataframe(df[['Year']+COLS],hide_index=True,use_container_width=True)

elif page=='Growth Rates':
    st.header('Annual Growth Rate Analysis')
    computed=df[['Year']+COLS].copy()
    computed[COLS]=df[COLS].pct_change()*100
    growth=reported_growth if is_reference else computed
    chart=growth.melt('Year',var_name='Indicator',value_name='Growth (%)').dropna()
    fig=px.line(chart,x='Year',y='Growth (%)',color='Indicator',markers=True,title='Year-on-Year Growth Rate Comparison',template='plotly_dark',color_discrete_sequence=['#64e4bb','#ffbe69','#78bfff','#db9bf0'])
    fig.add_hline(y=0,line_dash='dash',line_color='gray')
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(growth.style.format({c:'{:+.2f}%' for c in COLS}),use_container_width=True,hide_index=True)
    csv_button('Download growth rates',growth,'growth_rates.csv')

elif page=='Correlation & Climate':
    st.header('Pearson Correlation and Climate')
    available=COLS+[c for c in ['Rainfall','Temperature'] if c in df]
    corr=reported_corr.loc[available,available] if is_reference else df[available].corr()
    fig=px.imshow(corr,text_auto='.3f',zmin=-1,zmax=1,color_continuous_scale='RdBu_r',aspect='auto',title='Pearson Correlation Matrix (n = 5)');fig.update_layout(template='plotly_dark',height=510,paper_bgcolor='#172538',plot_bgcolor='#172538',font_color='#eef5ff')
    st.plotly_chart(fig,use_container_width=True)
    st.subheader('Production vs. Selected Indicators')
    for c in [x for x in ['Harvested Area','Yield','Farmgate Price','Rainfall','Temperature'] if x in df]:
        fig=px.scatter(df,x=c,y='Production',hover_data=['Year'],title=f'Production vs {c}',labels={'Production':'Production (MT)',c:f'{c} ({UNITS[c]})'},template='plotly_dark',color_discrete_sequence=['#64e4bb'])
        x=df[c].to_numpy(dtype=float); y=df.Production.to_numpy(dtype=float)
        if np.std(x)>0:
            slope,intercept=np.polyfit(x,y,1)
            xs=np.linspace(x.min(),x.max(),50)
            fig.add_trace(go.Scatter(x=xs,y=slope*xs+intercept,name='Linear fit',line=dict(color='#ffbe69')))
        st.plotly_chart(fig,use_container_width=True)
    st.subheader('Semestral Weather and Climate Classification')
    st.dataframe(sem,hide_index=True,use_container_width=True)
    for c in ['Rainfall','Temperature']:
        fig=px.line(sem.assign(Period=sem.Year.astype(str)+' S'+sem.Semester.astype(str)),x='Period',y=c,markers=True,title=f'Semestral {c}',template='plotly_dark',color_discrete_sequence=['#64e4bb'])
        st.plotly_chart(fig,use_container_width=True)

elif page=='Forecasting':
    st.header('Linear Trend Forecasting (2026–2030)')
    for c in COLS:
        st.plotly_chart(forecasting_chart(df,c,active_forecast),use_container_width=True)
    st.subheader('Projected Values')
    st.dataframe(active_forecast.style.format({c:'{:,.2f}' for c in COLS}),use_container_width=True,hide_index=True)
    csv_button('Download projections',active_forecast,'palay_forecast_2026_2030.csv')

elif page=='Model Evaluation':
    st.header('Model Evaluation')
    st.subheader('Linear Trend Model Performance')
    metrics=calculate_metrics(df)
    st.dataframe(metrics.style.format({'R²':'{:.4f}','MAE':'{:,.3f}','RMSE':'{:,.3f}','Slope/year':'{:+,.3f}'}),hide_index=True,use_container_width=True)
    colors=['#58dcb0','#61a5f7','#f7c574','#fa8c91']
    fig=go.Figure(go.Bar(
        x=metrics['Indicator'],y=metrics['R²'],
        marker_color=colors,
        hovertemplate='%{x}<br>R²: %{y:.6f}<extra></extra>'
    ))
    fig.update_layout(
        template='plotly_dark',height=380,
        paper_bgcolor='#102332',plot_bgcolor='#102332',
        font=dict(family='DM Sans',color='#f1f6ff',size=13),
        xaxis=dict(title='Indicator',showgrid=False,linecolor='#486176'),
        yaxis=dict(title='R²',range=[0,max(.75,float(metrics['R²'].max())+.07)],gridcolor='#2c4355',zeroline=False),
        margin=dict(l=35,r=20,t=35,b=35),showlegend=False,
        hoverlabel=dict(bgcolor='#20374a',font_color='#ffffff')
    )
    st.plotly_chart(fig,use_container_width=True)
    st.download_button('Download exact model evaluation',metrics.to_csv(index=False,float_format='%.10f').encode('utf-8-sig'),file_name='model_evaluation_exact.csv',mime='text/csv')

elif page=='Decision Support':
    st.header('Data-Driven Agricultural Forecasting Framework')
    steps=['Data Collection','Data Cleaning & Validation','Descriptive Statistics','Trend & Growth Analysis','Pearson Correlation','Linear Trend Forecasting','Model Evaluation','Visualization & Decision Support']
    st.markdown(' → '.join(f'**{x}**' for x in steps))
    st.subheader('Decision Support Outputs')
    p0,p1=float(df.Production.iloc[-1]),float(active_forecast.Production.iloc[-1])
    st.dataframe(pd.DataFrame([
        {'Output':'Production Outlook','Evidence':f'2025: {p0:,.2f} MT','Planning interpretation':'Monitor year-to-year changes in total palay supply'},
        {'Output':'Productivity Assessment','Evidence':f'2025 yield: {df.Yield.iloc[-1]:.2f} MT/ha','Planning interpretation':'Assess yield improvement alongside harvested area'},
        {'Output':'Climate Sensitivity Interpretation','Evidence':'Rainfall and temperature correlations (5 years)','Planning interpretation':'Rainfall and temperature context'},
        {'Output':'Future Supply Projection','Evidence':f'2030: {p1:,.2f} MT','Planning interpretation':'Support resource and production planning'}
    ]),hide_index=True,use_container_width=True)

st.divider()
