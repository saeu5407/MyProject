# 2019빅콘테스트 이노베이션 분야

> ### 미세먼지의 사회적 영향 분석 및 예측 모델링을 통한 BI 제시

> #### 환기타임 팀(이세욱, 박소이, 백가을)

> #### 미세먼지 상관분석을 통해 중장년층 인생 이모작 제시



<pre><code>
ipython
import cx_Oracle
import pandas as pd
import numpy as np
import os
from numpy import nan as NA
from pandas import Series, DataFrame # 함수명 그대로 사용하게 하는 방법
from datetime import datetime
from datetime import timedelta
os.putenv('NLS_LANG', 'KOREAN_KOREA.KO16MSWIN949') 
def f_sql(sql, ip='localhost',port='1521',user='scott',passwd='oracle', sid='orcl') :
    txt=user + '/' + passwd + "@" + ip + ':' + port + '/' + sid
    con=cx_Oracle.connect(txt)
    df=pd.read_sql(sql, con=con)
    return df
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
from IPython.display import display
import mglearn
</code></pre>



#### 기상 데이터

기상 데이터는 V10O1610252.csv 등 각각의 스테이션 별로 분당 기상 관측을 한 데이터이다.

널값으로 -999와 -9999가 있어서 제거를 했고, 

지점 : V10O1611289, 날짜 : 201808290738, pm2.5 : 4406 ㎍/㎥
지점 : V10O1611698, 날짜 : 201804031938, pm2.5 : 6900 ㎍/㎥
지점 : V10O1611097, 날짜 : 201807160538, pm2.5 : 4104 ㎍/㎥
지점 : V10O1611100, 날짜 : 201806160646, pm10 : 1057 ㎍/㎥ 
지점 : V10O1611100, 날짜 : 201808311238, pm2.5 : 3803 ㎍/㎥

에 이상치들이 발견이되어 전 시간대와 이후 시간대의 평균 값으로 대체하였다.

또한 일별, 주별, 월별로 데이터를 구분할 수 있게끔 컬럼을 추가하였고, 스테이션 별로 있던 데이터를 구별로 평균을 내었다.

> 구별로 낸 이유는 스테이션이 존재하지 않는 동이 있고, 사전조사 결과 구별로 미세먼지 등이 약간의 차이가 존재한다는 결론을 내려서 스테이션을 구별로 묶었다.

최종적으로 정리한 파일은 시간대별, 일별 두 가지로 저장하였다.

<pre><code>
v1 = pd.read_csv('V10O1610252.csv', encoding='cp949', na_values=[-999,-9999])
v2 = pd.read_csv('V10O1610546.csv', encoding='cp949', na_values=[-999,-9999])
v3 = pd.read_csv('V10O1610540.csv', encoding='cp949', na_values=[-999,-9999])
v4 = pd.read_csv('V10O1610542.csv', encoding='cp949', na_values=[-999,-9999])
v5 = pd.read_csv('V10O1610543.csv', encoding='cp949', na_values=[-999,-9999])
v6 = pd.read_csv('V10O1610544.csv', encoding='cp949', na_values=[-999,-9999])
v7 = pd.read_csv('V10O1610545.csv', encoding='cp949', na_values=[-999,-9999])
v8 = pd.read_csv('V10O1610567.csv', encoding='cp949', na_values=[-999,-9999])
v9 = pd.read_csv('V01o1610468.csv', encoding='cp949', na_values=[-999,-9999])
v10 = pd.read_csv('V10O1611289.csv', encoding='cp949', na_values=[-999,-9999])
v11 = pd.read_csv('V10O1611172.csv', encoding='cp949', na_values=[-999,-9999])
v12 = pd.read_csv('V10O1611634.csv', encoding='cp949', na_values=[-999,-9999])
v13 = pd.read_csv('V10O1611887.csv', encoding='cp949', na_values=[-999,-9999])
v14 = pd.read_csv('V10O1611639.csv', encoding='cp949', na_values=[-999,-9999])
v15 = pd.read_csv('V10O1611658.csv', encoding='cp949', na_values=[-999,-9999])
v16 = pd.read_csv('V10O1612113.csv', encoding='cp949', na_values=[-999,-9999])
v17 = pd.read_csv('V10O1611151.csv', encoding='cp949', na_values=[-999,-9999])
v18 = pd.read_csv('V10O1611145.csv', encoding='cp949', na_values=[-999,-9999])
v19 = pd.read_csv('V10O1611623.csv', encoding='cp949', na_values=[-999,-9999])
v20 = pd.read_csv('V10O1611750.csv', encoding='cp949', na_values=[-999,-9999])
v21 = pd.read_csv('V10O1611170.csv', encoding='cp949', na_values=[-999,-9999])
v22 = pd.read_csv('V10O1611684.csv', encoding='cp949', na_values=[-999,-9999])
v23 = pd.read_csv('V10O1611220.csv', encoding='cp949', na_values=[-999,-9999])
v24 = pd.read_csv('V10O1612106.csv', encoding='cp949', na_values=[-999,-9999])
v25 = pd.read_csv('V10O1611251.csv', encoding='cp949', na_values=[-999,-9999])
v26 = pd.read_csv('V10O1611173.csv', encoding='cp949', na_values=[-999,-9999])
v27 = pd.read_csv('V10O1611258.csv', encoding='cp949', na_values=[-999,-9999])
v28 = pd.read_csv('V10O1611255.csv', encoding='cp949', na_values=[-999,-9999])
v29 = pd.read_csv('V10O1611698.csv', encoding='cp949', na_values=[-999,-9999])
v30 = pd.read_csv('V10O1611722.csv', encoding='cp949', na_values=[-999,-9999])
v31 = pd.read_csv('V10O1611645.csv', encoding='cp949', na_values=[-999,-9999])
v32 = pd.read_csv('V10O1610610.csv', encoding='cp949', na_values=[-999,-9999])
v33 = pd.read_csv('V10O1610376.csv', encoding='cp949', na_values=[-999,-9999])
v34 = pd.read_csv('V10O1610293.csv', encoding='cp949', na_values=[-999,-9999])
v35 = pd.read_csv('V10O1610356.csv', encoding='cp949', na_values=[-999,-9999])
v36 = pd.read_csv('V10O1610616.csv', encoding='cp949', na_values=[-999,-9999])
v37 = pd.read_csv('V10O1610200.csv', encoding='cp949', na_values=[-999,-9999])
v38 = pd.read_csv('V10O1610643.csv', encoding='cp949', na_values=[-999,-9999])
v39 = pd.read_csv('V10O1610642.csv', encoding='cp949', na_values=[-999,-9999])
v40 = pd.read_csv('V10O1610297.csv', encoding='cp949', na_values=[-999,-9999])
v41 = pd.read_csv('V10O1610312.csv', encoding='cp949', na_values=[-999,-9999])
v42 = pd.read_csv('V10O1610102.csv', encoding='cp949', na_values=[-999,-9999])
v43 = pd.read_csv('V10O1610351.csv', encoding='cp949', na_values=[-999,-9999])
v44 = pd.read_csv('V10O1610629.csv', encoding='cp949', na_values=[-999,-9999])
v45 = pd.read_csv('V10O1610630.csv', encoding='cp949', na_values=[-999,-9999])
v46 = pd.read_csv('V10O1611104.csv', encoding='cp949', na_values=[-999,-9999])
v47 = pd.read_csv('V10O1611097.csv', encoding='cp949', na_values=[-999,-9999])
v48 = pd.read_csv('V10O1611100.csv', encoding='cp949', na_values=[-999,-9999])
v49 = pd.read_csv('V10O1612126.csv', encoding='cp949', na_values=[-999,-9999])
v50 = pd.read_csv('V10O1611102.csv', encoding='cp949', na_values=[-999,-9999])
v51 = pd.read_csv('V10O1611652.csv', encoding='cp949', na_values=[-999,-9999])
v52 = pd.read_csv('V10O1611150.csv', encoding='cp949', na_values=[-999,-9999])
v53 = pd.read_csv('V10O1611229.csv', encoding='cp949', na_values=[-999,-9999])
gisang = pd.concat([v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,
v23,v24,v25,v26,v27,v28,v29,v30,v31,v32,v33,v34,v35,v36,v37,v38,v39,v40,v41,v42,v43,v44,v45,v46,
v47,v48,v49,v50,v51,v52,v53])
gisang = gisang.iloc[:,:-1]
</code></pre>

<pre><code>
a1 = list(gisang.loc[(gisang.serial == 'V10O1611289') & (gisang.tm == 201808290737), 'pm25'])[0]
a2 = list(gisang.loc[(gisang.serial == 'V10O1611289') & (gisang.tm == 201808290739), 'pm25'])[0]
a3 = a1 + a2 / 2
gisang.loc[(gisang.serial == 'V10O1611289') & (gisang.tm == 201808290738), 'pm25'] = a3
b1 = list(gisang.loc[(gisang.serial == 'V10O1611698') & (gisang.tm == 201804031937), 'pm25'])[0]
>b2 = list(gisang.loc[(gisang.serial == 'V10O1611698') & (gisang.tm == 201804031939), 'pm25'])[0]
>b3 = b1 + b2 / 2
gisang.loc[(gisang.serial == 'V10O1611698') & (gisang.tm == 201804031938), 'pm25'] = b3
c1 = list(gisang.loc[(gisang.serial == 'V10O1611097') & (gisang.tm == 201807160537), 'pm25'])[0]
c2 = list(gisang.loc[(gisang.serial == 'V10O1611097') & (gisang.tm == 201807160539), 'pm25'])[0]
c3 = c1 + c2 / 2
gisang.loc[(gisang.serial == 'V10O1611097') & (gisang.tm == 201807160538), 'pm25'] = c3
d1 = list(gisang.loc[(gisang.serial == 'V10O1611100') & (gisang.tm == 201806160645), 'pm10'])[0]
d2 = list(gisang.loc[(gisang.serial == 'V10O1611100') & (gisang.tm == 201806160647), 'pm10'])[0]
d3 = d1 + d2 / 2
gisang.loc[(gisang.serial == 'V10O1611100') & (gisang.tm == 201806160646), 'pm10'] = d3
e1 = list(gisang.loc[(gisang.serial == 'V10O1611100') & (gisang.tm == 201808311238), 'pm25'])[0]
e2 = list(gisang.loc[(gisang.serial == 'V10O1611100') & (gisang.tm == 201808311238), 'pm25'])[0]
e3 = e1 + e2 / 2
gisang.loc[(gisang.serial == 'V10O1611100') & (gisang.tm == 201808311238), 'pm25'] = e3
</code></pre>

<pre><code>
>import datetime
from dateutil.parser import parse
gisang['yyyymmdd'] = gisang.tm.astype('str').str[:8]
gisang['weekday'] = gisang.tm.astype('str').str[:8].map(lambda x : parse(x).weekday())
gisang['tm'] = gisang.tm.map(lambda x : datetime.datetime.strptime(str(x), "%Y%m%d%H%M"))#1.pm10, pm25는 -999값이 nan처리가 안되어있었다.
#2.필요없는 컬럼 제외
gisang.pm10 = gisang.pm10.replace(-999,np.nan)
gisang.pm25 = gisang.pm25.replace(-999,np.nan)
gisang = gisang.drop(['co2','vocs','flag'], axis=1)
gisang.to_csv('gisang_1.csv', index=False, encoding='cp949')
</code></pre>
<pre><code>
>gi = pd.read_csv('gisang_1.csv', encoding='cp949', parse_dates=[0])
l1=['V10O1610252',
'V10O1610546','V10O1610540','V10O1610542','V10O1610543','V10O1610544','V10O1610545','V10O1610567','V01o1610468','V10O1611289','V10O1611172','V10O1611634','V10O1611887','V10O1611639','V10O1611658','V10O1612113','V10O1611151','V10O1611145','V10O1611623','V10O1611750','V10O1611170','V10O1611684','V10O1611220','V10O1612106','V10O1611251','V10O1611173','V10O1611258','V10O1611255','V10O1611698','V10O1611722','V10O1611645']
l2=[]
for i in list(gi.serial) :
    if i in l1 :
        l2.append('종로구')
    else :
        l2.append('노원구')
l2
gi.serial = l2
</code></pre>

<pre><code>
>gi1 = pd.pivot_table(data=gi, index=['tm','serial'], aggfunc='mean').reset_index()
gi1.to_csv('gisang_hour.csv', encoding='cp949', index=False)
gi2 = pd.pivot_table(data=gi, index=['yyyymmdd','serial'], aggfunc='mean').reset_index()
gi2.to_csv('gisang_day.csv', encoding='cp949', index=False)
</code></pre>



#### 유동인구 데이터

유동인구 데이터는 노원_종로_FLOW_AGE_201804.csv 처럼 일자별 성별 연령별 데이터로 구성된 파일과, 노원_종로_FLOW_TIME_201804.csv 처럼 성별 연령별을 알 수 없지만 시간대별로 구성된 파일이 존재했다.

다만 다른 파일들이 시간대별로 존재하지 않고 성별 연령별로 구분이 되어서 AGE 파일들만 사용하게 되었다. 추가적으로 남자 연령 전체와 여자 연령 전체, 모든 인원 컬럼 3개를 추가하였다. 이후 기상데이터와 묶었다.

<pre><code>
>flow1804 = pd.read_csv('노원_종로_FLOW_AGE_201804.csv', sep='|', engine='python', parse_dates=[1])
flow1805 = pd.read_csv('노원_종로_FLOW_AGE_201805.csv', sep='|', engine='python', parse_dates=[1])
flow1806 = pd.read_csv('노원_종로_FLOW_AGE_201806.csv', sep='|', engine='python', parse_dates=[1])
flow1807 = pd.read_csv('노원_종로_FLOW_AGE_201807.csv', sep='|', engine='python', parse_dates=[1])
flow1808 = pd.read_csv('노원_종로_FLOW_AGE_201808.csv', sep='|', engine='python', parse_dates=[1])
flow1809 = pd.read_csv('노원_종로_FLOW_AGE_201809.csv', sep='|', engine='python', parse_dates=[1])
flow1810 = pd.read_csv('노원_종로_FLOW_AGE_201810.csv', sep='|', engine='python', parse_dates=[1])
flow1811 = pd.read_csv('노원_종로_FLOW_AGE_201811.csv', sep='|', engine='python', parse_dates=[1])
flow1812 = pd.read_csv('노원_종로_FLOW_AGE_201812.csv', sep='|', engine='python', parse_dates=[1])
flow1901 = pd.read_csv('노원_종로_FLOW_AGE_201901.csv', sep='|', engine='python', parse_dates=[1])
flow1902 = pd.read_csv('노원_종로_FLOW_AGE_201902.csv', sep='|', engine='python', parse_dates=[1])
flow1903 = pd.read_csv('노원_종로_FLOW_AGE_201903.csv', sep='|', engine='python', parse_dates=[1])
flow = pd.concat([flow1804,flow1805,flow1806,flow1807,flow1808,flow1809,flow1810,flow1811,flow1812,flow1901,flow1902,flow1903])
flow = flow.iloc[:,1:].drop('HDONG_NM', axis=1)
</code></pre>

<pre><code>
>flow['MAN_FLOW_SUM'] = flow.loc[:,['MAN_FLOW_POP_CNT_0004',
       'MAN_FLOW_POP_CNT_0509', 'MAN_FLOW_POP_CNT_1014',
       'MAN_FLOW_POP_CNT_1519', 'MAN_FLOW_POP_CNT_2024',
       'MAN_FLOW_POP_CNT_2529', 'MAN_FLOW_POP_CNT_3034',
       'MAN_FLOW_POP_CNT_3539', 'MAN_FLOW_POP_CNT_4044',
       'MAN_FLOW_POP_CNT_4549', 'MAN_FLOW_POP_CNT_5054',
       'MAN_FLOW_POP_CNT_5559', 'MAN_FLOW_POP_CNT_6064',
       'MAN_FLOW_POP_CNT_6569', 'MAN_FLOW_POP_CNT_70U']].apply('sum', axis=1)
flow['WMAN_FLOW_SUM'] = flow.loc[:,['WMAN_FLOW_POP_CNT_0004', 'WMAN_FLOW_POP_CNT_0509',
       'WMAN_FLOW_POP_CNT_1014', 'WMAN_FLOW_POP_CNT_1519',
       'WMAN_FLOW_POP_CNT_2024', 'WMAN_FLOW_POP_CNT_2529',
       'WMAN_FLOW_POP_CNT_3034', 'WMAN_FLOW_POP_CNT_3539',
       'WMAN_FLOW_POP_CNT_4044', 'WMAN_FLOW_POP_CNT_4549',
       'WMAN_FLOW_POP_CNT_5054', 'WMAN_FLOW_POP_CNT_5559',
       'WMAN_FLOW_POP_CNT_6064', 'WMAN_FLOW_POP_CNT_6569',
       'WMAN_FLOW_POP_CNT_70U']].apply('sum', axis=1)
flow['TOTAL_FLOW_SUM'] = flow.loc[:,['WMAN_FLOW_SUM','MAN_FLOW_SUM']].apply('sum', axis=1)
list(set(flow.HDONG_CD))
flow.HDONG_CD = flow.HDONG_CD.astype('str').str[:8].astype('int')
flow = flow.rename(columns={'STD_YMD':'yyyymmdd'})
</code></pre>

<pre><code>
>m1 = pd.merge(gisang, flow, on='yyyymmdd')
m2 = pd.pivot_table(data=m1, index=['yyyymmdd','serial', 'humi', 'noise', 'pm10', 'pm25', 'temp', 'weekday'], aggfunc='sum').drop('HDONG_CD', axis=1)
m1.to_csv('day_gi_ud_dong.csv', encoding='cp949', index=False)
m2.to_csv('day_gi_ud_gu.csv', encoding='cp949')
</code></pre>



#### 카드 데이터

카드매출 데이터를 나이별, 성별, 동별로 전처리를 했었는데

>1. 동별로 하면 널값이 너무 많음
>2. 주거지역, 상업지역, 대학가, 산간지역으로 나누고자 했으나 기준이 모호한 동이 많음
>3. 제공된 종로구 노원구 데이터는 대표적인 상업지역, 주거지역임

라는 이유로 구별로 변경했다.

<pre><code>
>a1 = pd.read_csv('day_gi_ud_dong.csv', encoding='cp949', parse_dates=[0])
card = pd.read_csv('CARD_SPENDING_190809.txt', sep = '\s+', parse_dates=[0])
card['mctsexage'] = card.MCT_CAT_CD.astype('str') + '/' + card.SEX_CD + '/' + card.AGE_CD.astype('str')
card['mctsex'] = card.MCT_CAT_CD.astype('str') + '/' + card.SEX_CD 
card['sexage'] = card.SEX_CD + '/' + card.AGE_CD.astype('str')
card['mctage'] = card.MCT_CAT_CD.astype('str') + '/' + card.AGE_CD.astype('str')
pd.pivot_table(data=card, index=['STD_DD', 'GU_CD'], columns=['MCT_CAT_CD','SEX_CD','AGE_CD','mctsexage','mctsex','sexage','mctage'],
values=['USE_CNT','USE_AMT'], aggfunc='sum')
b1 = pd.pivot_table(data=card, index=['STD_DD', 'GU_CD'], columns=['mctsex'],
values=['USE_CNT','USE_AMT'], aggfunc='sum')
c1=[]
for i in list(b1.columns.levels[0]) : 
   for j in list(b1.columns.levels[1]) :
      c1.append(i+j)
b1.columns= c1
b1 = b1.reset_index()
b2 = pd.pivot_table(data=card, index=['STD_DD', 'GU_CD'], columns=['mctage'],
values=['USE_CNT','USE_AMT'], aggfunc='sum')
c2=[]
for i in list(b2.columns.levels[0]) : 
   for j in list(b2.columns.levels[1]) :
      c2.append(i+j)
b2.columns= c2
b2 = b2.reset_index()
b3 = pd.pivot_table(data=card, index=['STD_DD', 'GU_CD'], columns=['mctsexage'],
values=['USE_CNT','USE_AMT'], aggfunc='sum')
c3=[]
for i in list(b3.columns.levels[0]) : 
   for j in list(b3.columns.levels[1]) :
      c3.append(i+j)
b3.columns= c3
b3 = b3.reset_index()
b4 = pd.pivot_table(data=card, index=['STD_DD', 'GU_CD'], columns=['MCT_CAT_CD'],
values=['USE_CNT','USE_AMT'], aggfunc='sum')
c4=[]
for i in list(b4.columns.levels[0]) : 
   for j in list(b4.columns.levels[1].astype('str')) :
      c4.append(i+j+'sum')
b4.columns= c4
b4 = b4.reset_index()
m1 = pd.merge(pd.merge(b1,b2, on=['STD_DD','GU_CD']), pd.merge(b3,b4, on=['STD_DD','GU_CD']), on=['STD_DD','GU_CD']).fillna(0)
m1.to_csv('card_all.csv', encoding='cp949', index=False)
</code></pre>



#### 추가

기상 데이터와 유동인구 등을 분석했었는데, 06시부터 23시까지 유동인구가 많은 시간대였고, 미세먼지는 시간대별로 차이가 약간은 있었지만 대부분 중국발 먼지가 많을 때가 많다보니 시간대별로 가공할 필요가 없었다. 그래서 유동인구만 06~23시만 추가하였다.

<pre><code>
>flow['0623_SUM'] = flow.iloc[:,8:].apply(sum, axis=1)
b1 = pd.read_csv('day_gi_ud_dong.csv', encoding='cp949', parse_dates=[0])
m1 = pd.merge(b1,flow, on=['yyyymmdd','HDONG_CD'])
m1.to_csv('gisang_ud_0623.csv', index=False, encoding='cp949')
a1 = pd.read_csv('gisang_ud_0623.csv', encoding='cp949', parse_dates=[0])
a2 = pd.pivot_table(data=a1.drop('HDONG_CD', axis=1), index=['yyyymmdd', 'serial', 'humi', 'noise', 'pm10', 'pm25', 'temp',
       'weekday'], aggfunc='sum').reset_index()
m1 = m1.rename(columns={'STD_DD':'yyyymmdd','GU_CD':'serial'})
m1.serial = m1.serial.replace([110,350],['종로구','노원구'])
m2 = pd.merge(a2, m1, on=['yyyymmdd','serial'])
m2.to_csv('merge_ud_card_pm.csv', index=False, encoding='cp949')
</code></pre>



#### 데이터 마이닝

프로젝트 당시에 정리한게 아니다보니 약간 누락된 코드가 있을수도 있다.
마이닝 중에 상관분석을 통해 미세먼지와 연관성이 있는 품목을 찾아보았다.

최종적으로 찾아낸 컬럼은 서적 매출 컬럼으로 45~50세 이상이 미세먼지와 양의 상관관계였다.

주중과 주말을 비교해봤을 때 모든 연령대가 주말에는 서적과 양의 상관관계였다. 하지만 주중의 경우 50세 이상만 양의 상관관계를 보였고 우리는 이 결과를
>1. 모든 사람들은 휴일에 미세먼지가 많을 때 서점에 간다.
>2. 50세 이상은 은퇴와 맞물린 연령대로 평일과 휴일이 비슷하다.
로 결론 지었다.

추가적인 자료 조사와 함께 우리는 50대 이상의 사람들이 평일에도 휴식을 취한다면 그들을 위한 인생 이모작을 제시하거나, 자기가 원하는 취미가 없는 사람들을 위한 취미를 찾아주는 플랫폼을 제시하기로 했다.

<pre><code>
>m2 = pd.read_csv('merge_ud_card_pm.csv', encoding='cp949', parse_dates=[0])
m3 = m2.set_index('yyyymmdd')
</code></pre>


###### 모든 카드 매출 마이닝 
<pre><code>
m4 = m3.loc[:, ['serial','pm10',
 'USE_AMT10sum','USE_AMT20sum','USE_AMT21sum','USE_AMT22sum',
 'USE_AMT30sum','USE_AMT31sum','USE_AMT32sum','USE_AMT33sum',
 'USE_AMT34sum','USE_AMT35sum','USE_AMT40sum','USE_AMT42sum',
 'USE_AMT43sum','USE_AMT44sum','USE_AMT50sum','USE_AMT52sum',
 'USE_AMT60sum','USE_AMT62sum','USE_AMT70sum',
 'USE_AMT71sum','USE_AMT80sum','USE_AMT81sum','USE_AMT92sum']]
corr = m4.loc[m4.serial=='종로구'].corr(method = 'pearson')
corr.iloc[0:1]
a1=corr.iloc[0:1]
corr = m4.loc[m4.serial=='노원구'].corr(method = 'pearson')
corr.iloc[0:1]
a2=corr.iloc[0:1]
m4 = m3.loc[:, ['serial','pm10',
 'USE_CNT10sum','USE_CNT20sum','USE_CNT21sum','USE_CNT22sum',
 'USE_CNT30sum','USE_CNT31sum','USE_CNT32sum','USE_CNT33sum',
 'USE_CNT34sum','USE_CNT35sum','USE_CNT40sum','USE_CNT42sum',
 'USE_CNT43sum','USE_CNT44sum','USE_CNT50sum','USE_CNT52sum',
 'USE_CNT60sum','USE_CNT62sum','USE_CNT70sum','USE_CNT71sum',
 'USE_CNT80sum','USE_CNT81sum','USE_CNT92sum']]
corr = m4.loc[m4.serial=='종로구'].corr(method = 'pearson')
corr.iloc[0:1]
a3=corr.iloc[0:1]
corr = m4.loc[m4.serial=='노원구'].corr(method = 'pearson')
corr.iloc[0:1]
a4=corr.iloc[0:1]
</code></pre>


###### 성별
<pre><code>
m4 = m3.loc[:, ['serial','pm10',
 'USE_CNT20/M','USE_CNT20/F','USE_CNT21/M','USE_CNT21/F',
 'USE_CNT32/M','USE_CNT32/F','USE_CNT33/M',
 'USE_CNT33/F','USE_CNT34/M','USE_CNT34/F','USE_CNT40/M',
 'USE_CNT40/F','USE_CNT42/M','USE_CNT42/F','USE_CNT50/M',
 'USE_CNT50/F','USE_CNT62/M','USE_CNT62/F','USE_CNT80/M','USE_CNT80/F',
]]
corr = m4.loc[m4.serial=='종로구'].corr(method = 'pearson')
corr.iloc[0:1]
b1=corr.iloc[0:1]
corr = m4.loc[m4.serial=='노원구'].corr(method = 'pearson')
corr.iloc[0:1]
b2=corr.iloc[0:1]
</code></pre>


###### 연령별
<pre><code>
m4 = m3.loc[:, ['serial','pm10',
 'USE_CNT50/20','USE_CNT50/25','USE_CNT50/30','USE_CNT50/35',
 'USE_CNT50/40','USE_CNT50/45','USE_CNT50/50','USE_CNT50/55',
 'USE_CNT50/60','USE_CNT50/65''USE_AMT50/20','USE_AMT50/25',
 'USE_AMT50/30','USE_AMT50/35','USE_AMT50/40','USE_AMT50/45',
 'USE_AMT50/50','USE_AMT50/55','USE_AMT50/60','USE_AMT50/65'
]]
corr = m4.loc[m4.serial=='종로구'].corr(method = 'pearson')
corr.iloc[0:1]
c50_0=corr.iloc[0:1]
corr = m4.loc[m4.serial=='노원구'].corr(method = 'pearson')
corr.iloc[0:1]
c50_1=corr.iloc[0:1]
</code></pre>


###### 주중 주말 비교
서적 데이터
<pre><code>
>m4 = m3.loc[m3.weekday.isin([1,2,3,4]), ['serial','pm10',
 'USE_CNT50/20',
 'USE_CNT50/25',
 'USE_CNT50/30',
 'USE_CNT50/35',
 'USE_CNT50/40',
 'USE_CNT50/45',
 'USE_CNT50/50',
 'USE_CNT50/55',
 'USE_CNT50/60',
 'USE_CNT50/65'
]]
corr = m4.loc[m4.serial=='종로구'].corr(method = 'pearson')
wd50_0 = corr.iloc[0:1,:]
wd50_0.to_csv('wd50_0.csv', encoding='cp949', index=False)
corr = m4.loc[m4.serial=='노원구'].corr(method = 'pearson')
wd50_1 = corr.iloc[0:1,:]
wd50_1.to_csv('wd50_1.csv', encoding='cp949', index=False)
m4 = m3.loc[m3.weekday.isin([5,6]), ['serial','pm10',
 'USE_CNT50/20',
 'USE_CNT50/25',
 'USE_CNT50/30',
 'USE_CNT50/35',
 'USE_CNT50/40',
 'USE_CNT50/45',
 'USE_CNT50/50',
 'USE_CNT50/55',
 'USE_CNT50/60',
 'USE_CNT50/65'
]]
corr = m4.loc[m4.serial=='종로구'].corr(method = 'pearson')
we50_0 = corr.iloc[0:1,:]
we50_0.to_csv('we50_0.csv', encoding='cp949', index=False)
corr = m4.loc[m4.serial=='노원구'].corr(method = 'pearson')
we50_1 = corr.iloc[0:1,:]
we50_1.to_csv('we50_1.csv', encoding='cp949', index=False)
</code></pre>


###### 레저업소 데이터
<pre><code>
m4 = m3.loc[m3.weekday.isin([1,2,3,4]), ['serial','pm10',
 'USE_CNT21/20',
 'USE_CNT21/25',
 'USE_CNT21/30',
 'USE_CNT21/35',
 'USE_CNT21/40',
 'USE_CNT21/45',
 'USE_CNT21/50',
 'USE_CNT21/55',
 'USE_CNT21/60',
 'USE_CNT21/65'
]]
corr = m4.loc[m4.serial=='종로구'].corr(method = 'pearson')
wd21_0 = corr.iloc[0:1,:]
wd21_0.to_csv('wd21_0.csv', encoding='cp949', index=False)
corr = m4.loc[m4.serial=='노원구'].corr(method = 'pearson')
wd21_1 = corr.iloc[0:1,:]
wd21_1.to_csv('wd21_1.csv', encoding='cp949', index=False)
m4 = m3.loc[m3.weekday.isin([5,6]), ['serial','pm10',
 'USE_CNT21/20',
 'USE_CNT21/25',
 'USE_CNT21/30',
 'USE_CNT21/35',
 'USE_CNT21/40',
 'USE_CNT21/45',
 'USE_CNT21/50',
 'USE_CNT21/55',
 'USE_CNT21/60',
 'USE_CNT21/65'
]]
corr = m4.loc[m4.serial=='종로구'].corr(method = 'pearson')
we21_0 = corr.iloc[0:1,:]
we21_0.to_csv('we21_0.csv', encoding='cp949', index=False)
corr = m4.loc[m4.serial=='노원구'].corr(method = 'pearson')
we21_1 = corr.iloc[0:1,:]
we21_1.to_csv('we21_1.csv', encoding='cp949', index=False)
</code></pre>


#### 서적매출 회귀 모델

상관분석 뿐만이 아니라 정확한 인과관계 파악을 위해 회귀 모델을 만들었다.
강수량 등의 컬럼을 구해 추가하고, 목표인 50~65대 유동인구 컬럼 추가,
temp2로 온도 컬럼을 조금 더 설명력이 높게 변환했다 temp의 경우 낮거나 높은 온도에와 평균적인 온도가 차이가 났는데. abs를 통해 직선 형태로 변경했다.

<pre><code>
m2 = pd.read_csv('merge_ud_card_pm.csv', encoding='cp949', parse_dates=[0])
m2['USE_CNT50/40U'] = m2.loc[:,['USE_CNT50/45','USE_CNT50/50','USE_CNT50/55','USE_CNT50/60','USE_CNT50/65']].apply(sum, axis=1)
m2['USE_AMT50/40U'] = m2.loc[:,['USE_AMT50/45','USE_AMT50/50','USE_AMT50/55','USE_AMT50/60','USE_AMT50/65']].apply(sum, axis=1)
m2.to_csv('reg_all.csv', encoding='cp949', index=False)
jb = pd.read_csv('reg_all.csv', encoding='cp949', parse_dates=['yyyymmdd'])
jb['shop'] = np.where(jb.serial=='종로구',324,125).astype('int')
l1 = []
for i in jb.temp :
    if i >= 18 :
        l1.append(i-18)
    else : 
        l1.append(abs(i-18))
jb['temp2'] = l1
jb['USE_CNT50/50U'] = jb.loc[:,['USE_CNT50/50','USE_CNT50/55','USE_CNT50/60','USE_CNT50/65']].apply(sum, axis=1)
jb['USE_AMT50/50U'] = jb.loc[:,['USE_AMT50/50','USE_AMT50/55','USE_AMT50/60','USE_AMT50/65']].apply(sum, axis=1)
gang = pd.read_csv('seoul_gangsu.csv', encoding='cp949', parse_dates=[1])
gang = gang.iloc[:, 1:]
gang.columns = ['yyyymmdd', 'water', 'wind']
ppap = pd.merge(jb, gang, on='yyyymmdd')
ppap['sum50u']=ppap.loc[:,['WMAN_FLOW_POP_CNT_5054','WMAN_FLOW_POP_CNT_5559','WMAN_FLOW_POP_CNT_6064','WMAN_FLOW_POP_CNT_6569','WMAN_FLOW_POP_CNT_70U',
'MAN_FLOW_POP_CNT_5054','MAN_FLOW_POP_CNT_5559','MAN_FLOW_POP_CNT_6064','MAN_FLOW_POP_CNT_6569','MAN_FLOW_POP_CNT_70U']].apply(sum,axis=1)
ppap.to_csv('reg_50.csv', encoding='cp949', index=False)
</code></pre>


#### 최종 모델
강수량, 유동인구, 습도 등의 데이터들을 유의미한 p-value를 만들지 못함
최종적으로 미세먼지와 온도2, shop의 수를 설명변수로 결정

<pre><code>
final = pd.read_csv('reg_50.csv', encoding='cp949', parse_dates=['yyyymmdd'])
final.weekday = final.weekday.astype('str')
import statsmodels.api as sm
final = final.rename(columns={'USE_AMT50/50U' : 'USE_AMT5050U', 'USE_CNT50/50U' : 'USE_CNT5050U'})
final2 = final.loc[final.weekday.isin(['1','2','3','4']),:]
reg = sm.OLS.from_formula('USE_CNT5050U~pm10+temp2+shop',final2).fit()
reg.summary()
</code></pre>


####  최종 결과
<pre><code>
                            OLS Regression Results
Dep. Variable:           USE_CNT5050U   R-squared:                       0.904
Model:                            OLS   Adj. R-squared:                  0.903
Method:                 Least Squares   F-statistic:                     1290.
Date:                Tue, 10 Sep 2019   Prob (F-statistic):          5.70e-209
Time:                        11:43:40   Log-Likelihood:                -3346.3
No. Observations:                 416   AIC:                             6701.
Df Residuals:                     412   BIC:                             6717.
Df Model:                           3
Covariance Type:            nonrobust
                 coef    std err          t      P>|t|      [0.025      0.975]
Intercept  -2855.8019    119.836    -23.831      0.000   -3091.368   -2620.236
pm10           6.4920      1.132      5.733      0.000       4.266       8.718
temp2         47.7047      6.456      7.390      0.000      35.015      60.395
shop          22.9206      0.373     61.387      0.000      22.187      23.655
Omnibus:                      154.293   Durbin-Watson:                   1.944
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1267.847
Skew:                           1.349   Prob(JB):                    4.90e-276
Kurtosis:                      11.116   Cond. No.                         810.
</code></pre>


#### 최종 회귀식
######**USE_CNT5050U = -2855.8019 + 6.492pm10 + 47.7047temp2 + 22.9206shop**

최종적으로 미세먼지와 온도, shop의 수가 영향이 있다는 걸 알 수 있었다.
이 결론을 토대로 BI를 제시하기로 했다.
내용은 PPT에 있다.
