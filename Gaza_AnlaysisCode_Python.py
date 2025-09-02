# Gaza Period Comparison Code
# Tim Hoheneder, University of New Hampshire
# 16 March 2025

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
gaza_file= 'Gaza_NDBI-NDVI_AnalysisVals.csv'
#DataFrame Entry:
df= pd.read_csv(gaza_file, delimiter=",", comment='#', header=0, 
                na_values= [-9999, "NA", "NaN", "NAN"], encoding= 'unicode_escape')
   
#%% Generate Plot and Correlation Metrics:    

# Create Scatter Plot Function
def ScatterPlot(df1, df2, colour, colour2, Label1, Label2):
    #Line of Best Fit:
    coefficients = np.polyfit(df1, df2, 1)
    line_of_best_fit = np.poly1d(coefficients)
    #Generate Scatter Plot:  
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.scatter(df1, df2, color=colour, s=10, alpha=0.75)
    ax1.plot(df1, line_of_best_fit(df1), color=colour2)
    #Plot Formatting:
    ax1.set_xlim(-0.35, 0.35)
    ax1.set_ylim(-0.35, 0.35)
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
    ax1.text(0.225, 0.275, f'Rho: {correlation}\np-Value: {p_value}',
             fontsize=12, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8))

#Display Scatterplot:
ScatterPlot(df['NDVI_PreWarCeasefire_MeanDiff'], df['NDBI_PreWarCeasefire_MeanDiff'], 'navy', 'maroon', 
            'NDVI Change: Pre-War to Ceasefire', 'NDBI Change: Pre-War to Ceasefire')

#%% Scatter Plot for Z-Scores:  

# Create Scatter Plot Function
def ScatterPlot(df1, df2, colour, colour2, Label1, Label2):
    #Line of Best Fit:
    coefficients = np.polyfit(df1, df2, 1)
    line_of_best_fit = np.poly1d(coefficients)
    #Generate Scatter Plot:  
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.scatter(df1, df2, color=colour, s=10, alpha=0.75)
    ax1.plot(df1, line_of_best_fit(df1), color=colour2)
    #Plot Formatting:
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
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
    ax1.text(-3, -3, f'Rho: {correlation}\np-Value: {p_value}',
             fontsize=12, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8))

#Display Scatterplot:
ScatterPlot(df['NDVI_PreWarCeasefire_ZScoreDiff'], df['NDBI_PreWarCeasefire_ZScoreDiff'], 'navy', 'maroon', 
            'NDVI Z-Score Change: Pre-War to Ceasefire', 'NDBI Z-Score Change: Pre-War to Ceasefire')

ScatterPlot(df['NDVI_Ceasefire_ZScore'], df['NDBI_Ceasefire_ZScore'], 'navy', 'maroon', 
            'NDVI Z-Score: Ceasefire', 'NDBI Z-Score: Ceasefire')

#%% NDVI Annual Figures: 

#Dataframe Manipulation:
#Create Copy of Daatframe for NDVI: 
df_NDVI = df    
#Drop All Values Below 0: 
df_NDVI = df_NDVI[df_NDVI['NDVI_Mean_2020'] >= 0]
df_NDVI = df_NDVI[df_NDVI['NDVI_Mean_2021'] >= 0]
df_NDVI = df_NDVI[df_NDVI['NDVI_Mean_2022'] >= 0]
df_NDVI = df_NDVI[df_NDVI['NDVI_Mean_2023'] >= 0]
df_NDVI = df_NDVI[df_NDVI['NDVI_Mean_2024'] >= 0]
df_NDVI = df_NDVI[df_NDVI['NDVI_Mean_2025'] >= 0]    

# Create Scatter Plot Function
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
    ax1.set_xlim(-0.05, 1)
    ax1.set_ylim(-0.05, 1)
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
    ax1.text(0.8, 0.15, f'Rho: {correlation}\np-Value: {p_value}',
             fontsize=12, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8))

#Display Scatterplot:
ScatterPlot(df_NDVI['NDVI_Mean_2023'], df_NDVI['NDVI_Mean_2024'], 'navy', 'maroon', 'NDVI: 2023', 'NDVI: 2024')
ScatterPlot(df_NDVI['NDVI_Mean_2024'], df_NDVI['NDVI_Mean_2025'], 'navy', 'maroon', 'NDVI: 2024', 'NDVI: 2025')
ScatterPlot(df_NDVI['NDVI_Mean_2023'], df_NDVI['NDVI_Mean_2025'], 'navy', 'maroon', 'NDVI: 2023', 'NDVI: 2025')
ScatterPlot(df_NDVI['NDVI_Mean_2020'], df_NDVI['NDVI_Mean_2025'], 'navy', 'maroon', 'NDVI: 2020', 'NDVI: 2025')

#%% Analytics of Annual NDVI Datasets: 

#Mean Values of Annual Columns: 
#Generate Years: 
years = np.array([2020, 2021, 2022, 2023, 2024, 2025])
#Generate Mean NDVI Values:
NDVI_Mean_2020 = df_NDVI['NDVI_Mean_2020'].mean()
NDVI_Mean_2021 = df_NDVI['NDVI_Mean_2021'].mean()
NDVI_Mean_2022 = df_NDVI['NDVI_Mean_2022'].mean()
NDVI_Mean_2023 = df_NDVI['NDVI_Mean_2023'].mean()
NDVI_Mean_2024 = df_NDVI['NDVI_Mean_2024'].mean()
NDVI_Mean_2025 = df_NDVI['NDVI_Mean_2025'].mean()
NDVI_Means = [NDVI_Mean_2020, NDVI_Mean_2021, NDVI_Mean_2022, NDVI_Mean_2023, NDVI_Mean_2024, NDVI_Mean_2025]
#Line of Best Fit Equation:
slope, intercept, r_value, p_value, std_err = stats.linregress(years, NDVI_Means)
trend_line = slope * years + intercept

#Plot NDVI Trend:
fig, ax1 = plt.subplots(figsize=(7, 5))
#Plot Formatting:
ax1.set_ylim(-0.01, 0.275)
ax1.plot(years, NDVI_Means, marker='o', linestyle='-', color='k')
ax1.plot(years, trend_line, linestyle='--', color='red')
ax1.set_xlabel('Analysis Year: 15 Jan - 15 Mar', fontweight='bold')
ax1.set_ylabel('NDVI Mean', fontweight='bold')
ax1.grid(True)

#%% Bar Plot for Data: 

#Load Data: 
df = pd.read_csv("Gaza_LandCover_AnalysisVals.csv")
categories = df['ClassName']
pre_conflict = df['PC_Pct']
post_conflict = df['PostC_Pct']
x = range(len(categories))
width = 0.4

#Plotting:
plt.figure(figsize=(10, 5))
bars1 = plt.bar([i - width/2 for i in x], pre_conflict, width, label='Pre-Conflict', color='darkgrey')
bars2 = plt.bar([i + width/2 for i in x], post_conflict, width, label='Post-Conflict', color='grey')
# Add text annotations above bars
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

#Formatting:
plt.xticks(ticks=x, labels=categories, rotation=45, ha='right')
plt.ylabel('Land Area (%)', fontweight='bold')
plt.xlabel('Land Cover Class', fontweight='bold')
plt.ylim([0, 65])
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()

#%%

#Load Dataframe:
df = pd.read_csv('Gaza_LandCover_NDBI_AnalysisVals.csv')
date_columns = df.columns.drop('LandCover')
dates = pd.to_datetime(date_columns, format='%Y_%m_%d')
ndvi_data = df.set_index('LandCover')
ndvi_data.columns = dates
ndvi_data = ndvi_data[~ndvi_data.index.isin(['Water'])]
landcover_colors = {'Cropland': (0.941, 0.588, 1.0), 'Built-Up': (0.980, 0.0, 0.0), 'Tree': (0.0, 0.392, 0.0),
    'Shrubland': (1.0, 0.733, 0.133), 'Grassland': (1.0, 1.0, 0.298),'Bare': (0.706, 0.706, 0.706)}

#Transpose Data:
ndvi_long = ndvi_data.T.reset_index()
ndvi_long = pd.melt(ndvi_long, id_vars='index', var_name='LandCover', value_name='NDVI')
ndvi_long.rename(columns={'index': 'Date'}, inplace=True)
ndvi_long['Month'] = ndvi_long['Date'].dt.month

#Deseasonalize NDVI: 
monthly_means = ndvi_long.groupby(['LandCover', 'Month'])['NDVI'].mean().reset_index()
monthly_means.rename(columns={'NDVI': 'LongTerm_MonthlyMean'}, inplace=True)
ndvi_long = ndvi_long.merge(monthly_means, on=['LandCover', 'Month'], how='left')
ndvi_long['NDVI_Deseasonalized'] = ndvi_long['NDVI'] - ndvi_long['LongTerm_MonthlyMean']

#Aggregate to Monthly Means: 
ndvi_long['MonthStart'] = ndvi_long['Date'].dt.to_period('M').dt.to_timestamp()
monthly_deseasonalized = ndvi_long.groupby(['MonthStart', 'LandCover'])['NDVI_Deseasonalized'].mean().reset_index()
plot_data = monthly_deseasonalized.pivot(index='MonthStart', columns='LandCover', values='NDVI_Deseasonalized')

#Generate Time Series:
start_date = datetime(2023, 10, 7)
end_date = datetime(2025, 1, 15)
plt.figure(figsize=(14, 6))
for col in plot_data.columns:
    color = landcover_colors.get(col, (0.5, 0.5, 0.5)) 
    plt.plot(plot_data.index, plot_data[col], label=col, color=color)
plt.axhline(y=0, color='k')
plt.axvline(start_date, color='grey', linewidth=0.5)
plt.axvline(end_date, color='grey', linewidth=0.5)
plt.axvspan(start_date, end_date, color='grey', alpha=0.25, label='Conflict Period')
#Formatting:
plt.xlabel('Date', fontweight='bold', fontsize=14)
plt.ylabel('Deseasonalized NDBI Value', fontweight='bold', fontsize=14)
plt.grid(True)
plt.xticks(rotation=45)
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xlim(pd.to_datetime('2020-12-25'), pd.to_datetime('2025-03-01'))
#Legend:
plt.legend(frameon=True)
plt.tight_layout()

#%% Non-Deseasonalized Time Series:

# Load DataFrame
df = pd.read_csv('Gaza_LandCover_NDBI_AnalysisVals.csv')
date_columns = df.columns.drop('LandCover')
dates = pd.to_datetime(date_columns, format='%Y_%m_%d')
ndvi_data = df.set_index('LandCover')
ndvi_data.columns = dates
ndvi_data = ndvi_data[~ndvi_data.index.isin(['Water'])]
landcover_colors = {'Cropland': (0.941, 0.588, 1.0), 'Built-Up': (0.980, 0.0, 0.0), 'Tree': (0.0, 0.392, 0.0),
    'Shrubland': (1.0, 0.733, 0.133), 'Grassland': (1.0, 1.0, 0.298),'Bare': (0.706, 0.706, 0.706)}

# Transpose Data
ndvi_long = ndvi_data.T.reset_index()
ndvi_long = pd.melt(ndvi_long, id_vars='index', var_name='LandCover', value_name='NDVI')
ndvi_long.rename(columns={'index': 'Date'}, inplace=True)

# Aggregate to Monthly Means (no deseasonalization)
ndvi_long['MonthStart'] = ndvi_long['Date'].dt.to_period('M').dt.to_timestamp()
monthly_raw = ndvi_long.groupby(['MonthStart', 'LandCover'])['NDVI'].mean().reset_index()
plot_data = monthly_raw.pivot(index='MonthStart', columns='LandCover', values='NDVI')

# Generate Time Series
start_date = datetime(2023, 10, 7)
end_date = datetime(2025, 1, 15)
plt.figure(figsize=(14, 6))
for col in plot_data.columns:
    color = landcover_colors.get(col, (0.5, 0.5, 0.5))
    plt.plot(plot_data.index, plot_data[col], label=col, color=color)
plt.axhline(y=0, color='k')
plt.axvline(start_date, color='grey', linewidth=0.5)
plt.axvline(end_date, color='grey', linewidth=0.5)
plt.axvspan(start_date, end_date, color='grey', alpha=0.25, label='Conflict Period')

# Formatting
plt.xlabel('Date', fontweight='bold', fontsize=14)
plt.ylabel('NDBI Value', fontweight='bold', fontsize=14)
plt.grid(True)
plt.xticks(rotation=45)
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xlim(pd.to_datetime('2020-12-25'), pd.to_datetime('2025-03-01'))

# Legend
plt.legend(frameon=True)
plt.tight_layout()
plt.show()

#%% Land Cover Change Gridded Plot: 

#Load Data and Reclassify Water Class: 
df = pd.read_csv('Gaza_SamplePoints_LCVals2.csv')
df = df[(df['LC_PreConflict'] != 6) & (df['LC_PostConflict'] != 6)]
df['LC_PreConflict'] = df['LC_PreConflict'].replace({7: 6})
df['LC_PostConflict'] = df['LC_PostConflict'].replace({7: 6})

#Define LC Labels: 
lc_labels_display = {0: 'Tree Cover', 1: 'Shrubland', 2: 'Grassland', 3: 'Cropland',
    4: 'Built-Up', 5: 'Bare', 6: 'Water'}
counts = df.groupby(['LC_PreConflict', 'LC_PostConflict']).size().reset_index(name='count')

#Generate Scatter Plot: 
plt.figure(figsize=(10, 10))
plt.scatter(counts['LC_PreConflict'], counts['LC_PostConflict'], s=counts['count']*25, alpha=0.5, 
            color='grey', edgecolors='k')
#Formatting:
valid_classes= sorted(lc_labels_display.keys())
plt.xlabel('Pre-Conflict Land Cover Class', fontweight='bold', fontsize=14)
plt.ylabel('Post-Conflict Land Cover Class', fontweight='bold', fontsize=14)
plt.xticks(ticks=valid_classes, labels=[lc_labels_display[i] for i in valid_classes], rotation=45, ha='center')
plt.yticks(ticks=valid_classes, labels=[lc_labels_display[i] for i in valid_classes])
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')
#Annotations for Higher Counts:
for _, row in counts.iterrows():
    if row['count'] > 5:
        plt.text(row['LC_PreConflict'], row['LC_PostConflict'], str(row['count']),
                 ha='center', va='center', fontsize=10, color='black')
plt.tight_layout()

#%% End of Code