import numpy as np
import pandas as pd
from copy import copy, deepcopy
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from matplotlib.backends.backend_pdf import PdfPages

dfheight=pd.read_csv('../data/raw/Results from Val_Roseg_Timelapse in µm per sec.csv')
dfdates=pd.read_csv('../data/raw/image_dates.csv',parse_dates=['time'])
dfw=pd.read_csv('../data/raw/water_dates.csv',parse_dates=['time'])
dfheight=dfheight['Y']
dfheight=3000-dfheight
dfheight=dfheight-1164 #Subtracting additional length from camera
dfheight*=6/170 #170 pixels=6ft
dfh=pd.DataFrame(columns=['time','height','water'])
dfh['time']=dfdates['time']
dfh['water']=dfw['water']
dfh['time']=pd.to_datetime(dfh['time'],format='%Y:%m:%d %H:%M:%S')
dfh['height']=dfheight.apply(lambda x: float(x)*0.3048)
dfh=dfh.set_index('time')
dfh.to_csv('../data/interim/height_data.csv')

df=pd.read_csv('../data/raw/Roseg excel.csv')
df.columns=['doy','time','temp','rh','ws']
df['time']=df['time']/100
d=df['time']
d=d.tolist()
c=df['doy'].tolist()
for i in range(0,df.shape[0]):
    if c[i]>300:
        df.loc[i,'time']=datetime(2016, 1, 1) + timedelta(days=c[i]-1,hours=d[i])
    else:
        df.loc[i,'time']=datetime(2017, 1, 1) + timedelta(days=c[i]-1,hours=d[i])
df=df.set_index('time')

for time in dfh.index:
    h=dfh.loc[str(time),'height']
    w=dfh.loc[str(time),'water']
    time=time.replace(hour=time.hour+1,minute=0,second=0)
    df.loc[str(time),'height']=h
    df.loc[str(time),'water']=w
df=df.dropna()
df.to_csv('../data/interim/roseg_measurements.csv')

pp = PdfPages('../data/processed/plots.pdf')
df['temp'].plot(figsize=(12,8), grid=True)
df['height'].plot(figsize=(12,8), grid=True)
df['water'].plot(figsize=(12,8), grid=True)
pp.savefig()
plt.clf()
pp.close()
