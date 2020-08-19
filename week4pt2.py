# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 19:20:10 2020

@author: user
"""

import pandas as pd
birddata=pd.read_csv(r'C:\Users\user\Using python for research\week4\bird_tracking.csv')

import matplotlib.pyplot as plt

import numpy as np

ix=birddata.bird_name=='Eric'
x,y=birddata.longitude[ix],birddata.latitude[ix]
plt.figure(figsize=(7,7))
plt.plot(x,y,'.')
plt.figure(figsize=(7,7))
bird_names=pd.unique(birddata.bird_name)

plt.figure(figsize=(7,7))
for bird_name in bird_names:
    ix=birddata.bird_name==bird_name
    x,y=birddata.longitude[ix],birddata.latitude[ix]
    plt.plot(x,y,'.',label=bird_name)
plt.xlabel("Longtitude")
plt.ylabel('Latitude')
plt.legend(loc='lower right')
plt.savefig('3traj.pdf')

ix=birddata.bird_name=='Eric'
speed=birddata.speed_2d[ix]
plt.hist(speed)

np.isnan(speed).any()
np.sum(np.isnan(speed))
ind=np.isnan(speed)
plt.hist(speed[~ind],bins=np.linspace(0,30,20),density=True)
plt.xlabel('2D speed m/s')
plt.ylabel('Frequency');


birddata.speed_2d.plot(kind='hist', range=[0,30]) 

import datetime
time_1=datetime.datetime.today()
date_str=birddata.date_time[0]
datetime.datetime.strptime(date_str[:-3],'%Y-%m-%d %H:%M:%S')
timestamps=[]
for k in range(len(birddata)):
    timestamps.append(datetime.datetime.strptime\
    (birddata.date_time.iloc[k][:-3],'%Y-%m-%d %H:%M:%S'))

birddata['timestamp']=pd.Series(timestamps,index=birddata.index)



