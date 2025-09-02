# Gaza Per Image Time Series Creation Code
# Tim Hoheneder, University of New Hampshire
# 15 May 2025

#%% Import Libraries: 
    
#Import Libraries:
import numpy as np                    
import pandas as pd
import matplotlib as mpl              
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from decimal import Decimal
import scipy.stats as stats
import matplotlib.dates as mdates
import datetime
from calendar import month_abbr
import calendar
import re
from datetime import datetime

#High-Res Figures:
mpl.rcParams['figure.dpi'] = 300

#%% Import Dataset:

#Import CSV Files:
df = pd.read_csv('NDVI_Per_Point_Per_OverpassDate_WideFormat.csv')
df = df.set_index(df.columns[0])
df_t = df

#%% Dataframe Manipulation: 
    
# Strip "_NDVI" from column names and drop non-date rows
df_t.index = df_t.index.str.replace('_NDVI', '', regex=False)
df_t = df_t[df_t.index.str.match(r'^\d{4}_\d{2}_\d{2}$')]
df_t.index = pd.to_datetime(df_t.index, format='%Y_%m_%d')

#Daily Mean NDVI DataFrame:
df_avg = pd.DataFrame({'Date': df_t.index,'MeanNDVI': df_t.mean(axis=1)})

#Deseasonalize and Long-Term Means:
df_avg['MonthNum'] = df_avg['Date'].dt.month
# Group by month and compute mean and stdev
monthly_stats = df_avg.groupby('MonthNum')['MeanNDVI'].agg(['mean', 'std']).reset_index()
monthly_stats.columns = ['MonthNum', 'LongTerm_MeanNDVI', 'Monthly_StDev_NDVI']
df_avg = pd.merge(df_avg, monthly_stats, on='MonthNum', how='left')
df_avg['NDVI_Deseasonalized'] = df_avg['MeanNDVI'] - df_avg['LongTerm_MeanNDVI']

#%% Create Aggregated Monthly and Yearly DataFrames

#Create Month and Year Columns: 
df_avg['Year'] = df_avg['Date'].dt.year
df_avg['Month'] = df_avg['Date'].dt.to_period('M')

#Monthly Aggregated DataFrame:
df_monthly = df_avg.groupby('Month').agg({'MeanNDVI': 'mean','NDVI_Deseasonalized': 'mean', 'Monthly_StDev_NDVI': 
                                          'mean'}).reset_index()
df_monthly['Month'] = df_monthly['Month'].dt.to_timestamp()

#Yearly Aggregated DataFrame:
df_yearly = df_avg.groupby('Year').agg({'MeanNDVI': 'mean', 'NDVI_Deseasonalized': 'mean', 'Monthly_StDev_NDVI': 
                                        'mean'}).reset_index()
df_yearly['YearDate'] = pd.to_datetime(df_yearly['Year'].astype(str) + '-01-01')

#Split Data into Conflict Periods:
#Define Cutoffs:
cutoff_1 = pd.to_datetime('2023-10-07')
cutoff_2 = pd.to_datetime('2025-01-15')
df_before = df_avg[df_avg['Date'] < cutoff_1].copy()
df_during = df_avg[(df_avg['Date'] >= cutoff_1) & (df_avg['Date'] <= cutoff_2)].copy()
df_before['Period'] = 'Pre-Conflict Period'
df_during['Period'] = 'Conflict Period'



#%% Violin Plot (Raw NDVI) with Mean and Median using Matplotlib

#Plot Violin Plot of NDVI Raw Values:
fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(y=0, color='k', alpha=0.666)
parts = ax.violinplot([ndvi_before, ndvi_during], showmeans=True, showmedians=True)
median_line, = ax.plot([], [], color='navy', label='Median')
mean_line, = ax.plot([], [], color='maroon', label='Mean')
#Formatting:
ax.grid(True)
ax.set_ylim(-0.05, 0.5)
ax.set_xticks([1, 2])
ax.set_xticklabels(['Pre-Conflict Period', 'Conflict Period'], fontsize=12, fontweight='bold')
ax.set_ylabel('NDVI Value', fontsize=12, fontweight='bold')
for pc in parts['bodies']:
    pc.set_facecolor('grey')
    pc.set_edgecolor('k')
    pc.set_alpha(0.25)
    pc.set_color('black')
parts['cbars'].set_edgecolor('k') 
parts['cmins'].set_edgecolor('k')
parts['cmaxes'].set_edgecolor('k')
parts['cmeans'].set_edgecolor('maroon')
parts['cmedians'].set_edgecolor('navy')
#Legend:
ax.legend(handles=[mean_line, median_line], loc='upper right', frameon=True, facecolor='whitesmoke', edgecolor='k', framealpha=1)
#ANOVA:
f_stat, p_value = stats.f_oneway(ndvi_before, ndvi_during)
print("One-Way ANOVA:")
print(f"F-Statistic = {f_stat}")
print(f"p-value = {p_value}")
if p_value < 0.05:
    print("Significant Difference Between Periods")
else:
    print("No Significant Difference Between Periods")
stat_text = f"F = {f_stat:.2f}\nP = {p_value:.3e}"
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='k')
ax.text(0.4, 0.975, stat_text, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left',
        fontsize=11, bbox=props)

#%% Create Time Series of NDVI: 

#Define conflict Dates:
start_date = pd.to_datetime('2023-10-07')
end_date = pd.to_datetime('2025-01-15')  

#Daily:
#Plot NDVI Trend:
fig, ax1 = plt.subplots(figsize=(20, 5))
ax1.axhline(y=0, color='k')
ax1.axvline(start_date, color='grey',linewidth= 0.5)
ax1.axvline(end_date, color='grey', linewidth= 0.5)
ax1.errorbar(df_avg['Date'], df_avg['NDVI_Deseasonalized'], yerr=df_avg['Monthly_StDev_NDVI'], fmt='o-', 
            color='dimgrey', ecolor='gray', capsize=3, label='Deseasonalized NDVI')
#Line of Best Fit:
x = mdates.date2num(df_avg['Date'])
slope, intercept = np.polyfit(x, df_avg['NDVI_Deseasonalized'], 1)
trendline = slope * x + intercept
ax1.plot(df_avg['Date'], trendline, linestyle='--', color='maroon', alpha=0.75, label='Trendline')
#Formatting:
ax1.axvspan(start_date, end_date, color='grey', alpha=0.25, label='Conflict Period')  
ax1.set_xlabel('Analysis Date', fontweight='bold')
ax1.set_ylabel('NDVI Mean Value', fontweight='bold')
ax1.set_ylim(-0.2, 0.2)
ax1.set_xlim(pd.to_datetime('2020-12-15'), pd.to_datetime('2025-03-01'))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
ax1.grid(True)
fig.autofmt_xdate()
#Legend:
ax1.legend(loc='upper left', frameon=False)

#Plot Monthly Time Series:
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.axhline(y=0, color='k')
ax1.axvline(start_date, color='grey',linewidth= 0.5)
ax1.axvline(end_date, color='grey', linewidth= 0.5)
ax1.errorbar(df_monthly['Month'], df_monthly['NDVI_Deseasonalized'], yerr=df_monthly['Monthly_StDev_NDVI'], fmt='o-', 
            color='dimgrey', ecolor='gray', capsize=3, label='Deseasonalized NDVI')
#Line of Best Fit:
x = mdates.date2num(df_monthly['Month'])
slope, intercept = np.polyfit(x, df_monthly['NDVI_Deseasonalized'], 1)
trendline = slope * x + intercept
ax1.plot(df_monthly['Month'], trendline, linestyle='--', color='maroon', alpha=0.75, label='Trendline')
#Formatting:
ax1.axvspan(start_date, end_date, color='grey', alpha=0.25, label='Conflict Period')  
ax1.set_xlabel('Analysis Month', fontweight='bold')
ax1.set_ylabel('NDVI Monthly Mean Value', fontweight='bold')
ax1.set_ylim(-0.2, 0.2)
ax1.set_xlim(pd.to_datetime('2020-12-15'), pd.to_datetime('2025-03-01'))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
ax1.grid(True)
fig.autofmt_xdate()
#Legend:
ax1.legend(loc='upper left', frameon=False)

#Plot Yearly Time Series: 
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.axhline(y=0, color='k')
ax1.axvline(start_date, color='grey',linewidth= 0.5)
ax1.axvline(end_date, color='grey', linewidth= 0.5)
ax1.errorbar(df_yearly['YearDate'], df_yearly['NDVI_Deseasonalized'], yerr=df_yearly['Monthly_StDev_NDVI'], fmt='o-', 
            color='dimgrey', ecolor='gray', capsize=3, label='Deseasonalized NDVI')
#Line of Best Fit:
x = mdates.date2num(df_yearly['YearDate'])
slope, intercept = np.polyfit(x, df_yearly['NDVI_Deseasonalized'], 1)
trendline = slope * x + intercept
ax1.plot(df_yearly['YearDate'], trendline, linestyle='--', color='maroon', alpha=0.75, label='Trendline')
#Formatting:
ax1.axvspan(start_date, end_date, color='grey', alpha=0.25, label='Conflict Period')  
ax1.set_xlabel('Analysis Year', fontweight='bold')
ax1.set_ylabel('NDVI Yearly Mean Value', fontweight='bold')
ax1.set_ylim(-0.2, 0.2)
ax1.set_xlim(pd.to_datetime('2020-12-15'), pd.to_datetime('2025-03-01'))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
ax1.grid(True)
fig.autofmt_xdate()
#Legend:
ax1.legend(loc='upper left', frameon=False)

#%% Create Stacked Aggregated Time Series Plot: 

#Stacked: 
#Raw Data: 
#Plot Yearly Time Series: 
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.axhline(y=0, color='k')
ax1.axvline(start_date, color='grey',linewidth= 0.5)
ax1.axvline(end_date, color='grey', linewidth= 0.5)
ax1.plot(df_yearly['YearDate'], df_yearly['MeanNDVI'], label='Yearly Mean NDVI', color='darkgreen', linestyle='-')
ax1.plot(df_monthly['Month'], df_monthly['MeanNDVI'], label='Monthly Mean NDVI', color='navy', linestyle='-')
ax1.plot(df_avg['Date'], df_avg['MeanNDVI'], label='Daily NDVI', color='maroon', linestyle='-')
#Formatting:
ax1.axvspan(start_date, end_date, color='grey', alpha=0.25, label='Conflict Period')  
ax1.set_xlabel('Analysis Date', fontweight='bold')
ax1.set_ylabel('NDVI Value', fontweight='bold')
ax1.set_ylim(-0.01, 0.5)
ax1.set_xlim(pd.to_datetime('2020-12-15'), pd.to_datetime('2025-03-01'))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
ax1.grid(True)
fig.autofmt_xdate()
#Legend:
ax1.legend(loc='lower left', frameon=False)
#Deseasonalized Data: 
#Plot Yearly Time Series: 
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.axhline(y=0, color='k')
ax1.axvline(start_date, color='grey',linewidth= 0.5)
ax1.axvline(end_date, color='grey', linewidth= 0.5)
ax1.plot(df_yearly['YearDate'], df_yearly['NDVI_Deseasonalized'], label='Yearly Mean NDVI', color='darkgreen', linestyle='-')
ax1.plot(df_monthly['Month'], df_monthly['NDVI_Deseasonalized'], label='Monthly Mean NDVI', color='navy', linestyle='-')
ax1.plot(df_avg['Date'], df_avg['NDVI_Deseasonalized'], label='Daily NDVI', color='maroon', linestyle='-')
#Formatting:
ax1.axvspan(start_date, end_date, color='grey', alpha=0.25, label='Conflict Period')  
ax1.set_xlabel('Analysis Date', fontweight='bold')
ax1.set_ylabel('Deseasonalized NDVI Value', fontweight='bold')
ax1.set_ylim(-0.15, 0.1)
ax1.set_xlim(pd.to_datetime('2020-12-15'), pd.to_datetime('2025-03-01'))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
ax1.grid(True)
fig.autofmt_xdate()
#Legend:
ax1.legend(loc='lower left', frameon=False)

#Combined:
fig, ax1 = plt.subplots(2, 1, figsize=(12, 10))
#Plot Yearly Time Series: 
ax1[0].axhline(y=0, color='k')
ax1[0].axvline(start_date, color='grey',linewidth= 0.5)
ax1[0].axvline(end_date, color='grey', linewidth= 0.5)
ax1[0].plot(df_yearly['YearDate'], df_yearly['MeanNDVI'], label='Yearly Mean NDVI', color='darkgreen', linestyle='-')
ax1[0].plot(df_monthly['Month'], df_monthly['MeanNDVI'], label='Monthly Mean NDVI', color='navy', linestyle='-')
ax1[0].plot(df_avg['Date'], df_avg['MeanNDVI'], label='Daily NDVI', color='maroon', linestyle='-')
#Formatting:
ax1[0].axvspan(start_date, end_date, color='grey', alpha=0.25, label='Conflict Period')  
ax1[0].set_xlabel('Analysis Date', fontweight='bold')
ax1[0].set_ylabel('NDVI Value', fontweight='bold')
ax1[0].set_ylim(-0.05, 0.4)
ax1[0].set_xlim(pd.to_datetime('2020-12-15'), pd.to_datetime('2025-03-01'))
ax1[0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
ax1[0].grid(True)
#Deseasonalized Data: 
#Plot Yearly Time Series: 
ax1[1].axhline(y=0, color='k')
ax1[1].axvline(start_date, color='grey',linewidth= 0.5)
ax1[1].axvline(end_date, color='grey', linewidth= 0.5)
ax1[1].plot(df_yearly['YearDate'], df_yearly['NDVI_Deseasonalized'], label='Yearly Mean NDVI', color='darkgreen', linestyle='-')
ax1[1].plot(df_monthly['Month'], df_monthly['NDVI_Deseasonalized'], label='Monthly Mean NDVI', color='navy', linestyle='-')
ax1[1].plot(df_avg['Date'], df_avg['NDVI_Deseasonalized'], label='Daily NDVI', color='maroon', linestyle='-')
#Formatting:
ax1[1].axvspan(start_date, end_date, color='grey', alpha=0.25, label='Conflict Period')  
ax1[1].set_xlabel('Analysis Date', fontweight='bold')
ax1[1].set_ylabel('Deseasonalized NDVI Value', fontweight='bold')
ax1[1].set_ylim(-0.15, 0.1)
ax1[1].set_xlim(pd.to_datetime('2020-12-15'), pd.to_datetime('2025-03-01'))
ax1[1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
ax1[1].grid(True)
fig.autofmt_xdate()
#Legend:
ax1[1].legend(loc='lower left', frameon=False)

#%% Repeat for NDBI: 

#Import CSV Files:
df = pd.read_csv('NDBI_Per_Point_Per_OverpassDate_WideFormat.csv')
df = df.set_index(df.columns[0])
df_t = df.transpose()

#%% Dataframe Manipulation: 

# Strip "_NDBI" from column names and drop non-date rows
df_t.index = df_t.index.str.replace('_NDBI', '', regex=False)
df_t = df_t[df_t.index.str.match(r'^\d{4}_\d{2}_\d{2}$')]
df_t.index = pd.to_datetime(df_t.index, format='%Y_%m_%d')

#Daily Mean NDBI DataFrame:
df_avg = pd.DataFrame({'Date': df_t.index,'MeanNDBI': df_t.mean(axis=1)})

#Deseasonalize and Long-Term Means:
df_avg['MonthNum'] = df_avg['Date'].dt.month
# Group by month and compute mean and stdev
monthly_stats = df_avg.groupby('MonthNum')['MeanNDBI'].agg(['mean', 'std']).reset_index()
monthly_stats.columns = ['MonthNum', 'LongTerm_MeanNDBI', 'Monthly_StDev_NDBI']
df_avg = pd.merge(df_avg, monthly_stats, on='MonthNum', how='left')
df_avg['NDBI_Deseasonalized'] = df_avg['MeanNDBI'] - df_avg['LongTerm_MeanNDBI']

#%% Create Aggregated Monthly and Yearly DataFrames

#Create Month and Year Columns: 
df_avg['Year'] = df_avg['Date'].dt.year
df_avg['Month'] = df_avg['Date'].dt.to_period('M')

#Monthly Aggregated DataFrame:
df_monthly = df_avg.groupby('Month').agg({'MeanNDBI': 'mean','NDBI_Deseasonalized': 'mean', 'Monthly_StDev_NDBI': 
                                          'mean'}).reset_index()
df_monthly['Month'] = df_monthly['Month'].dt.to_timestamp()

#Yearly Aggregated DataFrame:
df_yearly = df_avg.groupby('Year').agg({'MeanNDBI': 'mean', 'NDBI_Deseasonalized': 'mean', 'Monthly_StDev_NDBI': 
                                        'mean'}).reset_index()
df_yearly['YearDate'] = pd.to_datetime(df_yearly['Year'].astype(str) + '-01-01')

#Split Data into Conflict Periods:
#Define Cutoffs:
cutoff_1 = pd.to_datetime('2023-10-07')
cutoff_2 = pd.to_datetime('2025-01-15')
df_before = df_avg[df_avg['Date'] < cutoff_1].copy()
df_during = df_avg[(df_avg['Date'] >= cutoff_1) & (df_avg['Date'] <= cutoff_2)].copy()
df_before['Period'] = 'Pre-Conflict Period'
df_during['Period'] = 'Conflict Period'

#Extract NDVI:
ndvi_before = df_before['MeanNDBI'].dropna()
ndvi_during = df_during['MeanNDBI'].dropna()
ndbi_meandiff = ndvi_during - ndvi_before
df_ndbi = pd.DataFrame(ndbi_meandiff, columns=['MeanNDBI'])

#%% Violin Plot (Raw NDVI) with Mean and Median using Matplotlib

#Plot Violin Plot of NDVI Raw Values:
fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(y=0, color='k', alpha=0.666)
parts = ax.violinplot([ndvi_before, ndvi_during], showmeans=True, showmedians=True)
median_line, = ax.plot([], [], color='navy', label='Median')
mean_line, = ax.plot([], [], color='maroon', label='Mean')
#Formatting:
ax.grid(True)
ax.set_ylim(-0.2, 0.2)
ax.set_xticks([1, 2])
ax.set_xticklabels(['Pre-Conflict Period', 'Conflict Period'], fontsize=12, fontweight='bold')
ax.set_ylabel('NDBI Value', fontsize=12, fontweight='bold')
for pc in parts['bodies']:
    pc.set_facecolor('grey')
    pc.set_edgecolor('k')
    pc.set_alpha(0.25)
    pc.set_color('black')
parts['cbars'].set_edgecolor('k') 
parts['cmins'].set_edgecolor('k')
parts['cmaxes'].set_edgecolor('k')
parts['cmeans'].set_edgecolor('maroon')
parts['cmedians'].set_edgecolor('navy')
#Legend:
ax.legend(handles=[mean_line, median_line], loc='upper right', frameon=True, facecolor='whitesmoke', edgecolor='k', framealpha=1)
#ANOVA:
f_stat, p_value = stats.f_oneway(ndvi_before, ndvi_during)
print("One-Way ANOVA:")
print(f"F-Statistic = {f_stat}")
print(f"p-value = {p_value}")
if p_value < 0.05:
    print("Significant Difference Between Periods")
else:
    print("No Significant Difference Between Periods")
stat_text = f"F = {f_stat:.2f}\nP = {p_value:.3e}"
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='k')
ax.text(0.4, 0.975, stat_text, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left',
        fontsize=11, bbox=props)

#%% Create Time Series of NDBI:  

#Raw Data:
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.axhline(y=0, color='k')
ax1.axvline(start_date, color='grey', linewidth=0.5)
ax1.axvline(end_date, color='grey', linewidth=0.5)
ax1.plot(df_yearly['YearDate'], df_yearly['MeanNDBI'], label='Yearly Mean NDBI', color='darkgreen')
ax1.plot(df_monthly['Month'], df_monthly['MeanNDBI'], label='Monthly Mean NDBI', color='navy')
ax1.plot(df_avg['Date'], df_avg['MeanNDBI'], label='Daily NDBI', color='maroon')
ax1.axvspan(start_date, end_date, color='grey', alpha=0.25, label='Conflict Period')  
ax1.set_xlabel('Analysis Date', fontweight='bold')
ax1.set_ylabel('NDBI Value', fontweight='bold')
ax1.set_ylim(-0.01, 0.5)
ax1.set_xlim(pd.to_datetime('2020-12-15'), pd.to_datetime('2025-03-01'))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
ax1.grid(True)
fig.autofmt_xdate()
ax1.legend(loc='lower left', frameon=False)

#Deseasonalized:
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.axhline(y=0, color='k')
ax1.axvline(start_date, color='grey', linewidth=0.5)
ax1.axvline(end_date, color='grey', linewidth=0.5)
ax1.plot(df_yearly['YearDate'], df_yearly['NDBI_Deseasonalized'], label='Yearly Mean NDBI', color='darkgreen')
ax1.plot(df_monthly['Month'], df_monthly['NDBI_Deseasonalized'], label='Monthly Mean NDBI', color='navy')
ax1.plot(df_avg['Date'], df_avg['NDBI_Deseasonalized'], label='Daily NDBI', color='maroon')
ax1.axvspan(start_date, end_date, color='grey', alpha=0.25, label='Conflict Period')  
ax1.set_xlabel('Analysis Date', fontweight='bold')
ax1.set_ylabel('Deseasonalized NDBI Value', fontweight='bold')
ax1.set_ylim(-0.15, 0.1)
ax1.set_xlim(pd.to_datetime('2020-12-15'), pd.to_datetime('2025-03-01'))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
ax1.grid(True)
fig.autofmt_xdate()
ax1.legend(loc='lower left', frameon=False)

#Combined:  
fig, ax1 = plt.subplots(2, 1, figsize=(12, 10))
ax1[0].axhline(y=0, color='k')
ax1[0].axvline(start_date, color='grey', linewidth=0.5)
ax1[0].axvline(end_date, color='grey', linewidth=0.5)
ax1[0].plot(df_yearly['YearDate'], df_yearly['MeanNDBI'], label='Yearly Mean NDBI', color='darkgreen')
ax1[0].plot(df_monthly['Month'], df_monthly['MeanNDBI'], label='Monthly Mean NDBI', color='navy')
ax1[0].plot(df_avg['Date'], df_avg['MeanNDBI'], label='Daily NDBI', color='maroon')
ax1[0].axvspan(start_date, end_date, color='grey', alpha=0.25, label='Conflict Period')  
ax1[0].set_xlabel('Analysis Date', fontweight='bold')
ax1[0].set_ylabel('NDBI Value', fontweight='bold')
ax1[0].set_ylim(-0.2, 0.15)
ax1[0].set_xlim(pd.to_datetime('2020-12-15'), pd.to_datetime('2025-03-01'))
ax1[0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
ax1[0].grid(True)
ax1[1].axhline(y=0, color='k')
ax1[1].axvline(start_date, color='grey', linewidth=0.5)
ax1[1].axvline(end_date, color='grey', linewidth=0.5)
ax1[1].plot(df_yearly['YearDate'], df_yearly['NDBI_Deseasonalized'], label='Yearly Mean NDBI', color='darkgreen')
ax1[1].plot(df_monthly['Month'], df_monthly['NDBI_Deseasonalized'], label='Monthly Mean NDBI', color='navy')
ax1[1].plot(df_avg['Date'], df_avg['NDBI_Deseasonalized'], label='Daily NDBI', color='maroon')
ax1[1].axvspan(start_date, end_date, color='grey', alpha=0.25, label='Conflict Period')  
ax1[1].set_xlabel('Analysis Date', fontweight='bold')
ax1[1].set_ylabel('Deseasonalized NDBI Value', fontweight='bold')
ax1[1].set_ylim(-0.1, 0.15)
ax1[1].set_xlim(pd.to_datetime('2020-12-15'), pd.to_datetime('2025-03-01'))
ax1[1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
ax1[1].grid(True)
fig.autofmt_xdate()
ax1[1].legend(loc='upper left', frameon=False)

#%% Scatter Plot for Mean Difference: 

#NDVI:
#Load the CSV:
df = pd.read_csv('NDVI_Per_Point_Per_OverpassDate_WideFormat.csv')
df = df.rename(columns={col: col.replace('_NDVI', '') for col in df.columns if '_NDVI' in col})
date_columns = [col for col in df.columns if re.match(r'\d{4}_\d{2}_\d{2}', col)]
col_date_map = {col: datetime.strptime(col, '%Y_%m_%d') for col in date_columns}
#Define date Cutoffs: 
cutoff_1 = datetime(2023, 10, 7)
cutoff_2 = datetime(2025, 1, 15)
pre_cols = [col for col, date in col_date_map.items() if date < cutoff_1]
post_cols = [col for col, date in col_date_map.items() if cutoff_1 <= date <= cutoff_2]
#Compute Mean NDVI Values: 
df['NDVI_PreConflictMean'] = df[pre_cols].mean(axis=1, skipna=True)
df['NDVI_PostConflictMean'] = df[post_cols].mean(axis=1, skipna=True)
df_ndvi= df['NDVI_PostConflictMean'] - df['NDVI_PreConflictMean']
df_ndvi = pd.DataFrame(df_ndvi, columns=['MeanNDVI'])
df_ndvi['AbsChange'] = abs(df_ndvi['MeanNDVI'])
#Identify Less than 2.5% Change:
df_ndvi['NDVI_PercentChange'] = (df['NDVI_PostConflictMean'] - df['NDVI_PreConflictMean']) / df['NDVI_PreConflictMean'] * 100
df_smallchange_ndvi = df[abs(df['NDVI_PostConflictMean']-df['NDVI_PreConflictMean']) < 0.025]

#NDBI:
#Load the CSV:
df = pd.read_csv('NDBI_Per_Point_Per_OverpassDate_WideFormat.csv')
df = df.rename(columns={col: col.replace('_NDBI', '') for col in df.columns if '_NDBI' in col})
date_columns = [col for col in df.columns if re.match(r'\d{4}_\d{2}_\d{2}', col)]
col_date_map = {col: datetime.strptime(col, '%Y_%m_%d') for col in date_columns}
#Define Date Cutoffs: 
cutoff_1 = datetime(2023, 10, 7)
cutoff_2 = datetime(2025, 1, 15)
pre_cols = [col for col, date in col_date_map.items() if date < cutoff_1]
post_cols = [col for col, date in col_date_map.items() if cutoff_1 <= date <= cutoff_2]
#Compute Mean NDVI Values: 
df['NDBI_PreConflictMean'] = df[pre_cols].mean(axis=1, skipna=True)
df['NDBI_PostConflictMean'] = df[post_cols].mean(axis=1, skipna=True)
df_ndbi= df['NDBI_PostConflictMean'] - df['NDBI_PreConflictMean']
df_ndbi = pd.DataFrame(df_ndbi, columns=['MeanNDBI'])
df_ndbi['AbsChange'] = abs(df_ndbi['MeanNDBI'])
#Identify Less than 2.5% Change:
df_ndbi['NDBI_PercentChange'] = (df['NDBI_PostConflictMean'] - df['NDBI_PreConflictMean']) / df['NDBI_PreConflictMean'] * 100
df_smallchange_ndbi = df[abs(df['NDBI_PostConflictMean']-df['NDBI_PreConflictMean']) < 0.025]

#Both Indexes Minimal Change Calculation:
df_common_smallchange = pd.merge(df_smallchange_ndvi, df_smallchange_ndbi, on='ID', suffixes=('_NDVI', '_NDBI'))

#%% Scatter Plot: 
    
#Create Scatter Plot Function
def ScatterPlot(df1, df2, colour, colour2, Label1, Label2):
    #Line of Best Fit:
    coefficients = np.polyfit(df1, df2, 1)
    line_of_best_fit = np.poly1d(coefficients)
    #Generate Scatter Plot:  
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.scatter(df1, df2, color=colour, s=10, alpha=0.75)
    ax1.plot(df1, line_of_best_fit(df1), color=colour2)
    ax1.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=1)
    #Plot Formatting:
    ax1.set_xlim(-0.01, 0.4)
    ax1.set_ylim(-0.01, 0.4)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.75)
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.75)
    ax1.set_xlabel(Label1, fontweight='bold')
    ax1.set_ylabel(Label2, fontweight='bold')
    ax1.grid(True)
    #Spearman's Rho:
    correlation, p_value = spearmanr(df1, df2)
    print('Spearman Correlation Coefficient:', correlation)
    print('p-Value:', p_value)
    correlation = str(correlation)[:5]
    p_value = Decimal(p_value)
    p_value = "{:.2E}".format(p_value)
    ax1.text(0.075, 0.375, f'Rho: {correlation}\np-Value: {p_value}',
             fontsize=12, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8))

#Display Scatterplot:
ScatterPlot(df_ndvi['AbsChange'], df_ndbi['AbsChange'], 'navy', 'maroon', 
            'NDVI Absolute Mean Difference', 'NDBI Absolute Mean Difference')

#%% Z-Score Time Series Plotting: 

#Load Data:
df = pd.read_csv('NDVI_Per_Point_Per_OverpassDate_WideFormat.csv')
df = df.rename(columns={col: col.replace('_NDVI', '') for col in df.columns if '_NDVI' in col})

#Identify date columns and convert to datetime:
date_columns = [col for col in df.columns if re.match(r'\d{4}_\d{2}_\d{2}', col)]
col_date_map = {col: datetime.strptime(col, '%Y_%m_%d') for col in date_columns}
ndvi_values = df[date_columns]

#Z-score per point
z_scores = (ndvi_values - ndvi_values.mean(axis=1, skipna=True).values[:, None]) / ndvi_values.std(axis=1, skipna=True).values[:, None]
z_scores.columns = [col_date_map[col] for col in z_scores.columns]
z_scores_long = z_scores.T
z_scores_long.index.name = 'Date'
z_scores_long['DailyMeanZ'] = z_scores_long.mean(axis=1, skipna=True)
z_scores_long.reset_index(inplace=True)

#Deseasonalize Data:
z_scores_long['MonthNum'] = z_scores_long['Date'].dt.month
monthly_seasonal_mean = z_scores_long.groupby('MonthNum')['DailyMeanZ'].mean()
z_scores_long['DeseasonalizedZ'] = z_scores_long.apply(
    lambda row: row['DailyMeanZ'] - monthly_seasonal_mean[row['MonthNum']], axis=1)
#Daily:
daily_mean_z = z_scores_long[['Date', 'DailyMeanZ']]
daily_mean_z_deseasonalized = z_scores_long[['Date', 
                                             'DeseasonalizedZ']].rename(columns={'DeseasonalizedZ': 'DailyDeseasonalizedZ'})
#Monthly:
z_scores_long['Month'] = z_scores_long['Date'].dt.to_period('M')
monthly_mean_z = z_scores_long.groupby('Month')['DailyMeanZ'].mean().reset_index()
monthly_mean_z['Date'] = monthly_mean_z['Month'].dt.to_timestamp() + pd.DateOffset(days=14)
monthly_mean_z = monthly_mean_z[['Date', 'DailyMeanZ']].rename(columns={'DailyMeanZ': 'MonthlyMeanZ'})
monthly_mean_z_deseasonalized = z_scores_long.groupby('Month')['DeseasonalizedZ'].mean().reset_index()
monthly_mean_z_deseasonalized['Date'] = monthly_mean_z_deseasonalized['Month'].dt.to_timestamp() + pd.DateOffset(days=14)
monthly_mean_z_deseasonalized = monthly_mean_z_deseasonalized[['Date', 
                                                               'DeseasonalizedZ']].rename(columns={'DeseasonalizedZ': 'MonthlyDeseasonalizedZ'})
#Yearly:
z_scores_long['Year'] = z_scores_long['Date'].dt.year
yearly_mean_z = z_scores_long.groupby('Year')['DailyMeanZ'].mean().reset_index()
yearly_mean_z['Date'] = pd.to_datetime(yearly_mean_z['Year'].astype(str) + '-01-01')
yearly_mean_z = yearly_mean_z[['Date', 'DailyMeanZ']].rename(columns={'DailyMeanZ': 'YearlyMeanZ'})
yearly_mean_z_deseasonalized = z_scores_long.groupby('Year')['DeseasonalizedZ'].mean().reset_index()
yearly_mean_z_deseasonalized['Date'] = pd.to_datetime(yearly_mean_z_deseasonalized['Year'].astype(str) + '-01-01')
yearly_mean_z_deseasonalized = yearly_mean_z_deseasonalized[['Date', 
                                                             'DeseasonalizedZ']].rename(columns={'DeseasonalizedZ': 'YearlyDeseasonalizedZ'})

#Plotting:
start_date = datetime(2023, 10, 7)
end_date = datetime(2025, 1, 15)
#Create Plotting Area:
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
#Non-Deseasonalized:
ax[0].axhline(y=0, color='k')
ax[0].axvline(start_date, color='grey', linewidth=0.5)
ax[0].axvline(end_date, color='grey', linewidth=0.5)
ax[0].axvspan(start_date, end_date, color='grey', alpha=0.25, label='Conflict Period')
#Plot Time Series:
ax[0].plot(daily_mean_z_deseasonalized['Date'], daily_mean_z['DailyMeanZ'],
        label='Daily Z-Score', color='maroon', alpha=0.4)
ax[0].plot(monthly_mean_z_deseasonalized['Date'], monthly_mean_z['MonthlyMeanZ'],
        label='Monthly Z-Score', color='navy')
ax[0].plot(yearly_mean_z_deseasonalized['Date'], yearly_mean_z['YearlyMeanZ'],
        label='Yearly Z-Score', color='darkgreen', linewidth=2)
#Formatting:
ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
ax[0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax[0].set_xlabel('Date', fontweight='bold')
ax[0].set_ylabel('Mean NDVI Z-Score', fontweight='bold')
ax[0].legend()
ax[0].grid(True)
#Deseasonalized:
ax[1].axhline(y=0, color='k')
ax[1].axvline(start_date, color='grey', linewidth=0.5)
ax[1].axvline(end_date, color='grey', linewidth=0.5)
ax[1].axvspan(start_date, end_date, color='grey', alpha=0.25, label='Conflict Period')
#Plot Time Series:
ax[1].plot(daily_mean_z_deseasonalized['Date'], daily_mean_z_deseasonalized['DailyDeseasonalizedZ'],
        label='Daily Deseasonalized Z-Score', color='maroon', alpha=0.4)
ax[1].plot(monthly_mean_z_deseasonalized['Date'], monthly_mean_z_deseasonalized['MonthlyDeseasonalizedZ'],
        label='Monthly Deseasonalized Z-Score', color='navy')
ax[1].plot(yearly_mean_z_deseasonalized['Date'], yearly_mean_z_deseasonalized['YearlyDeseasonalizedZ'],
        label='Yearly Deseasonalized Z-Score', color='darkgreen', linewidth=2)
#Formatting:
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
ax[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax[1].set_xlabel('Date', fontweight='bold')
ax[1].set_ylabel('Deseasonalized Mean NDVI Z-Score', fontweight='bold')
ax[1].legend()
ax[1].grid(True)
fig.autofmt_xdate()
plt.tight_layout()

#%%Repeat for NDBI:

# Load Data:
df = pd.read_csv('NDBI_Per_Point_Per_OverpassDate_WideFormat.csv')
df = df.rename(columns={col: col.replace('_NDBI', '') for col in df.columns if '_NDBI' in col})
# Identify date columns and convert to datetime:
date_columns = [col for col in df.columns if re.match(r'\d{4}_\d{2}_\d{2}', col)]
col_date_map = {col: datetime.strptime(col, '%Y_%m_%d') for col in date_columns}
ndbi_values = df[date_columns]

# Z-score per point
z_scores = (ndbi_values - ndbi_values.mean(axis=1, skipna=True).values[:, None]) / ndbi_values.std(axis=1, skipna=True).values[:, None]
z_scores.columns = [col_date_map[col] for col in z_scores.columns]
z_scores_long = z_scores.T
z_scores_long.index.name = 'Date'
z_scores_long['DailyMeanZ'] = z_scores_long.mean(axis=1, skipna=True)
z_scores_long.reset_index(inplace=True)

# Deseasonalize Data:
z_scores_long['MonthNum'] = z_scores_long['Date'].dt.month
monthly_seasonal_mean = z_scores_long.groupby('MonthNum')['DailyMeanZ'].mean()
z_scores_long['DeseasonalizedZ'] = z_scores_long.apply(
    lambda row: row['DailyMeanZ'] - monthly_seasonal_mean[row['MonthNum']], axis=1)
# Daily:
daily_mean_z = z_scores_long[['Date', 'DailyMeanZ']]
daily_mean_z_deseasonalized = z_scores_long[['Date', 'DeseasonalizedZ']].rename(columns={'DeseasonalizedZ': 'DailyDeseasonalizedZ'})
# Monthly:
z_scores_long['Month'] = z_scores_long['Date'].dt.to_period('M')
monthly_mean_z = z_scores_long.groupby('Month')['DailyMeanZ'].mean().reset_index()
monthly_mean_z['Date'] = monthly_mean_z['Month'].dt.to_timestamp() + pd.DateOffset(days=14)
monthly_mean_z = monthly_mean_z[['Date', 'DailyMeanZ']].rename(columns={'DailyMeanZ': 'MonthlyMeanZ'})
monthly_mean_z_deseasonalized = z_scores_long.groupby('Month')['DeseasonalizedZ'].mean().reset_index()
monthly_mean_z_deseasonalized['Date'] = monthly_mean_z_deseasonalized['Month'].dt.to_timestamp() + pd.DateOffset(days=14)
monthly_mean_z_deseasonalized = monthly_mean_z_deseasonalized[['Date', 'DeseasonalizedZ']].rename(columns={'DeseasonalizedZ': 'MonthlyDeseasonalizedZ'})
# Yearly:
z_scores_long['Year'] = z_scores_long['Date'].dt.year
yearly_mean_z = z_scores_long.groupby('Year')['DailyMeanZ'].mean().reset_index()
yearly_mean_z['Date'] = pd.to_datetime(yearly_mean_z['Year'].astype(str) + '-01-01')
yearly_mean_z = yearly_mean_z[['Date', 'DailyMeanZ']].rename(columns={'DailyMeanZ': 'YearlyMeanZ'})
yearly_mean_z_deseasonalized = z_scores_long.groupby('Year')['DeseasonalizedZ'].mean().reset_index()
yearly_mean_z_deseasonalized['Date'] = pd.to_datetime(yearly_mean_z_deseasonalized['Year'].astype(str) + '-01-01')
yearly_mean_z_deseasonalized = yearly_mean_z_deseasonalized[['Date', 'DeseasonalizedZ']].rename(columns={'DeseasonalizedZ': 'YearlyDeseasonalizedZ'})

# Plotting:
start_date = datetime(2023, 10, 7)
end_date = datetime(2025, 1, 15)
# Create Plotting Area:
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
# Non-Deseasonalized:
ax[0].axhline(y=0, color='k')
ax[0].axvline(start_date, color='grey', linewidth=0.5)
ax[0].axvline(end_date, color='grey', linewidth=0.5)
ax[0].axvspan(start_date, end_date, color='grey', alpha=0.25, label='Conflict Period')
ax[0].plot(daily_mean_z_deseasonalized['Date'], daily_mean_z['DailyMeanZ'], label='Daily Z-Score', color='maroon', alpha=0.4)
ax[0].plot(monthly_mean_z_deseasonalized['Date'], monthly_mean_z['MonthlyMeanZ'], label='Monthly Z-Score', color='navy')
ax[0].plot(yearly_mean_z_deseasonalized['Date'], yearly_mean_z['YearlyMeanZ'], label='Yearly Z-Score', color='darkgreen', linewidth=2)
ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
ax[0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax[0].set_xlabel('Date', fontweight='bold')
ax[0].set_ylabel('Mean NDBI Z-Score', fontweight='bold')
ax[0].legend()
ax[0].grid(True)
# Deseasonalized:
ax[1].axhline(y=0, color='k')
ax[1].axvline(start_date, color='grey', linewidth=0.5)
ax[1].axvline(end_date, color='grey', linewidth=0.5)
ax[1].axvspan(start_date, end_date, color='grey', alpha=0.25, label='Conflict Period')
ax[1].plot(daily_mean_z_deseasonalized['Date'], daily_mean_z_deseasonalized['DailyDeseasonalizedZ'], label='Daily Deseasonalized Z-Score', color='maroon', alpha=0.4)
ax[1].plot(monthly_mean_z_deseasonalized['Date'], monthly_mean_z_deseasonalized['MonthlyDeseasonalizedZ'], label='Monthly Deseasonalized Z-Score', color='navy')
ax[1].plot(yearly_mean_z_deseasonalized['Date'], yearly_mean_z_deseasonalized['YearlyDeseasonalizedZ'], label='Yearly Deseasonalized Z-Score', color='darkgreen', linewidth=2)
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
ax[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax[1].set_xlabel('Date', fontweight='bold')
ax[1].set_ylabel('Deseasonalized Mean NDBI Z-Score', fontweight='bold')
ax[1].legend()
ax[1].grid(True)
fig.autofmt_xdate()
plt.tight_layout()

#%% End of Code