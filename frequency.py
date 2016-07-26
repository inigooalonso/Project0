# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:39:33 2016

@author: bdore
"""
#This script creates a dataframe that counts the number of events per unique device_id.
#The events are counted by hour of day and day of the week. There are 7x24 variables.
#The timestamp spans a total of 14 days so column "0_0" will count the number of events that happenend on sundays from 0h to 1am.
#I'm no python magician so feel free to improve the code or correct any bugs you see. If you encounter problems let me know.
#Since this takes a while to run I will upload the final csv to the repo. After reading the csv you must clean a duplicate 'device_id' column with zero values.
#IF you want to tune this code to get rid of that be my guest :)
#Bernardo Doré 

import pandas as pd
import numpy as np
import datetime

#Reads only timestamp and device_id columns from events.csv
events = pd.read_csv("../input/events.csv", usecols=['timestamp', 'device_id'])

#######################
#Extraction and formatting
#######################

#Converts timestamp to datetime object and format
events['timestamp'] = events['timestamp'].apply(lambda x :datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

#Extracts day of the week from timestamp
events['weekday'] = events['timestamp'].apply(lambda x : x.weekday())

#Extracts hour of the day from timestamp
events['hour'] = events['timestamp'].apply(lambda x : x.hour)

######################
#Number of events
#####################

#Get list of unique id's
unique_ids = list(events['device_id'].unique())

#Counts number of events for each device-id per hour per day.
#Stores result of each iteration in a dataframe then updates unique_df dict in {device_id:result} format.
#Takes a while to run.
unique_df = dict()
for i in range(0,len(unique_ids)):
    device_events = events.loc[events.device_id == unique_ids[i], :]
    df = pd.DataFrame(device_events[['weekday', 'hour', 'device_id']].groupby(['weekday', 'hour']).count())
    m = df.unstack()
    m = m.fillna(0)
    unique_df.update({unique_ids[i]:m})    

####################
#Final datarame
####################

#List of names following format 'Weekday_Hour'
weekdays = list(events['weekday'].unique())
hours = list(events['hour'].unique())
hours.sort()
weekdays.sort()
column_names = list()
column_names.append('device_id')

for i in weekdays:
    for j in hours:
        column_names.append(str(i) + "_" + str(j))

#Final dataframe
event_count = pd.DataFrame(columns=column_names, index=unique_ids)
event_count.rename = column_names

#Extracts relevant values from unique_df dict and stores in even_count.
#Takes a while to run.
for key, value in unique_df.items():   
    z = value
    for i in range(0,len(z)):        
        y = np.nonzero(z.iloc[i])        
        for j in range(0, len(y[0])):
            x = z.iat[i,y[0][j]]            
            week_hour = str(z.index[[i]][0])+"_"+str(z.columns[y[0][j]][1])            
            event_count.loc[key][week_hour] = x            

event_count = event_count.fillna(0)
event_count = event_count.apply(lambda x : x.astype(int))
event_count.to_csv('event_count.csv', index=True, index_label='device_id')
