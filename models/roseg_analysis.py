import numpy as np
import pandas as pd
from copy import copy, deepcopy
from matplotlib import pyplot as plt
from datetime import datetime, timedelta, date, time
from matplotlib.backends.backend_pdf import PdfPages
from  sklearn.linear_model import ElasticNet
model = ElasticNet()


df = pd.read_csv('../data/interim/roseg_measurements.csv', parse_dates=['time'])
del df['doy']
df['delta_time'] = (df['time']-df['time'].shift()).fillna(0)
df['delta_height'] = (df['height']-df['height'].shift()).fillna(0)
df['growth_rate'] = (df['delta_height']*100)/(df['delta_time'].apply(lambda x: x.total_seconds()/3600))
# mask=(df['growth_rate']>0) & (df['water']==0)
# df.loc[mask,'growth_rate']=0

dfdays = df[(df['delta_time']<timedelta(hours=18))
            & (df['water']==0)
            & (df['delta_time']>timedelta(hours=0))
            & ((df.time.apply(lambda x: x.time())>time(12))==True)]

dfnights= df[(df['delta_time']<timedelta(hours=18))
              & (df['water']==0)
              & (df['delta_time']>timedelta(hours=0))
              & ((df.time.apply(lambda x: x.time())<time(12))==True)]

dfdaysw = df[(df['delta_time']<timedelta(hours=18))
            & (df['water']==1)
            & (df['delta_time']>timedelta(hours=0))
            & ((df.time.apply(lambda x: x.time())>time(12))==True)]

dfnightsw = df[(df['delta_time']<timedelta(hours=18))
              & (df['water']==1)
              & (df['delta_time']>timedelta(hours=0))
              & ((df.time.apply(lambda x: x.time())<time(12))==True)]


pp = PdfPages('../data/processed/plots2.pdf')
fig=plt.figure(figsize=(14,6))
ax=fig.add_subplot(1,2,1)
dfnights.plot(x='temp', y='growth_rate', kind='scatter', title='Night freeze', ax=ax);
# add regression line
plt.plot(dfnights['temp'], np.poly1d(np.polyfit(dfnights['temp'], dfnights['growth_rate'], 1))(dfnights['temp']),
         color='r')
ax.set_xlabel('Temperature [C]')
ax.set_ylabel('Growth rate [cm/hour]')
ax=fig.add_subplot(1,2,2)
dfdays.plot(x='temp', y='growth_rate', kind='scatter', title='Day melt', ax=ax);
plt.plot(dfdays['temp'], np.poly1d(np.polyfit(dfdays['temp'], dfdays['growth_rate'], 1))(dfdays['temp']),
         color='r')
ax.set_xlabel('Temperature [C]')
ax.set_ylabel('Growth rate [cm/hour]')
fig.tight_layout()
pp.savefig()
plt.clf()

fig=plt.figure(figsize=(14,6))
ax=fig.add_subplot(1,2,1)
dfnightsw.plot(x='temp', y='growth_rate', kind='scatter', title='Night freeze', ax=ax);
# add regression line
plt.plot(dfnights['temp'], np.poly1d(np.polyfit(dfnights['temp'], dfnights['growth_rate'], 1))(dfnights['temp']),
         color='r')
ax.set_xlabel('Temperature [C]')
ax.set_ylabel('Growth rate [cm/hour]')
ax=fig.add_subplot(1,2,2)
dfdaysw.plot(x='temp', y='growth_rate', kind='scatter', title='Day melt', ax=ax);
plt.plot(dfdays['temp'], np.poly1d(np.polyfit(dfdays['temp'], dfdays['growth_rate'], 1))(dfdays['temp']),
         color='r')
ax.set_xlabel('Temperature [C]')
ax.set_ylabel('Growth rate [cm/hour]')
fig.tight_layout()
pp.savefig()
plt.clf()

dfb = pd.read_csv('../data/interim/roseg_data.csv', parse_dates=['time'])
dfb.set_index('time', inplace=True)

df = df[['time', 'height', 'growth_rate','water']].copy()

df=df.round(3)
df.to_csv('../data/processed/roseg_results.csv')


df['temp_12h_avg'] = df['time'].apply(lambda x: dfb[(dfb.index<=x) & (dfb.index>=(x-timedelta(hours=12)))]['temp'].mean())
df['temp_12h_count'] = df['time'].apply(lambda x: dfb[(dfb.index<=x) & (dfb.index>=(x-timedelta(hours=12)))]['temp'].count())

df['rh_12h_avg'] = df['time'].apply(lambda x: dfb[(dfb.index<=x) & (dfb.index>=(x-timedelta(hours=12)))]['rh'].mean())
df['rh_12h_count'] = df['time'].apply(lambda x: dfb[(dfb.index<=x) & (dfb.index>=(x-timedelta(hours=12)))]['rh'].count())

df['ws_12h_avg'] = df['time'].apply(lambda x: dfb[(dfb.index<=x) & (dfb.index>=(x-timedelta(hours=12)))]['ws'].mean())
df['ws_12h_count'] = df['time'].apply(lambda x: dfb[(dfb.index<=x) & (dfb.index>=(x-timedelta(hours=12)))]['ws'].count())

df = df[(df[[col for col in df.columns if col.endswith('count')]]>=5).all(axis='columns')]
df = df[[col for col in df.columns if not col.endswith('count')]]

df['type'] = df['time'].apply(lambda x: 'AM' if x.time()<time(12) else 'PM')
df.dropna(inplace=True)

var_dict = {'temp_12h_avg': 'Average temperature over previous 12 hours [C]',
            'rh_12h_avg': 'Average relative humidity over previous 12 hours [%]',
            'ws_12h_avg': 'Average wind speed over previous 12 hours'}
for var in var_dict:
    fig=plt.figure(figsize=(12,8))
    ax=fig.add_subplot(1,1,1)
    for mtype in [('AM', 'Night growth', 'b'), ('PM', 'Day melt', 'r')]:
        dftmp = df[(df['type']==mtype[0])&(df['water']==1)]
        dftmp.plot(x=var, y='growth_rate', kind='scatter', ax=ax, label=mtype[1], color=mtype[2])
        plt.plot(dftmp[var], np.poly1d(np.polyfit(dftmp[var], dftmp['growth_rate'], 1))(dftmp[var]),
                 color=mtype[2])
    plt.grid()
    ax.set_xlabel(var_dict[var])
    ax.set_ylabel('Growth rate [cm/hour]')
    pp.savefig()
    plt.clf()

#df=df[df['water']==1]
print(df['growth_rate'].max())

print('%d morning measurements' % len(df[df['type']=='AM']))
print('%d evening measurements' % len(df[df['type']=='PM']))

df['temp_12h_min'] = df['time'].apply(lambda x: dfb[(dfb.index<=x) & (dfb.index>=(x-timedelta(hours=12)))]['temp'].min())
df['temp_12h_max'] = df['time'].apply(lambda x: dfb[(dfb.index<=x) & (dfb.index>=(x-timedelta(hours=12)))]['temp'].max())
df['temp_12h_std'] = df['time'].apply(lambda x: dfb[(dfb.index<=x) & (dfb.index>=(x-timedelta(hours=12)))]['temp'].std())

df['rh_12h_min'] = df['time'].apply(lambda x: dfb[(dfb.index<=x) & (dfb.index>=(x-timedelta(hours=12)))]['rh'].min())
df['rh_12h_max'] = df['time'].apply(lambda x: dfb[(dfb.index<=x) & (dfb.index>=(x-timedelta(hours=12)))]['rh'].max())
df['rh_12h_std'] = df['time'].apply(lambda x: dfb[(dfb.index<=x) & (dfb.index>=(x-timedelta(hours=12)))]['rh'].std())

df['ws_12h_min'] = df['time'].apply(lambda x: dfb[(dfb.index<=x) & (dfb.index>=(x-timedelta(hours=12)))]['ws'].min())
df['ws_12h_max'] = df['time'].apply(lambda x: dfb[(dfb.index<=x) & (dfb.index>=(x-timedelta(hours=12)))]['ws'].max())
df['ws_12h_std'] = df['time'].apply(lambda x: dfb[(dfb.index<=x) & (dfb.index>=(x-timedelta(hours=12)))]['ws'].std())

model_am = ElasticNet()
model_pm = ElasticNet()
X_am = df[df['type']=='AM'][[col for col in df.columns if (col.startswith('temp')
                                                           or col.startswith('rh')
                                                           or col.startswith('ws')
                                                          )]]
y_am = df[df['type']=='AM']['growth_rate']
model_am.fit(X_am,y_am)
print('Morning observations: R-squared = %0.2f' % model_am.score(X_am,y_am))

X_pm = -df[df['type']=='PM'][[col for col in df.columns if (col.startswith('temp')
                                                            or col.startswith('rh')
                                                            or col.startswith('ws')
                                                           )]]
y_pm = df[df['type']=='PM']['growth_rate']
model_pm.fit(X_pm,y_pm)
print('Evening observations: R-squared = %0.2f' % model_pm.score(X_pm,y_pm))

fig=plt.figure(figsize=(16,8))

# getting unzipped lists of feature names and corresponding coefficients, sorted by absolute decreasing coefficient
features_am, weights_am = zip(*sorted(zip(X_am.columns, model_am.coef_), key=lambda x: abs(x[1]), reverse=True))
features_pm, weights_pm = zip(*sorted(zip(X_pm.columns, model_pm.coef_), key=lambda x: abs(x[1]), reverse=True))

ax=fig.add_subplot(1,2,1)
plt.bar(range(len(weights_am)), weights_am)
ax.set_ylabel('Coefficient in linear regression')
plt.xticks([x+0.5 for x in range(len(weights_am))], features_am, rotation='vertical')
ax.set_title('Night freeze ($R^2=%0.2f$)' % model_am.score(X_am,y_am), {'fontsize': 14, 'fontweight' : 'bold'})

ax=fig.add_subplot(1,2,2)
plt.bar(range(len(weights_pm)), weights_pm)
ax.set_ylabel('Coefficient in linear regression')
plt.xticks([x+0.5 for x in range(len(weights_pm))], features_pm, rotation='vertical')
ax.set_title('Day melt ($R^2=%0.2f$)' % model_pm.score(X_pm,y_pm), {'fontsize': 14, 'fontweight' : 'bold'})

plt.tight_layout()
pp.savefig()
plt.clf()
pp.close()
