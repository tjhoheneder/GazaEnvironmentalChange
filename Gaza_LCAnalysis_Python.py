# Gaza Land Cover Classes Gridded Scatter Plots Code
# Tim Hoheneder, University of New Hampshire
# 23 June 2025

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
import rasterio

#High-Res Figures:
mpl.rcParams['figure.dpi'] = 300

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
plt.figure(figsize=(12, 12))
plt.scatter(counts['LC_PreConflict'], counts['LC_PostConflict'], s=counts['count']*100, alpha=0.4, 
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
    if row['count'] >= 1:
        plt.text(row['LC_PreConflict'], row['LC_PostConflict'], str(row['count']),
                 ha='center', va='center', fontsize=11, color='black', fontweight='bold')
plt.tight_layout()

#%% Generate Overall Raster Gridded Scatter Plot:  

#Load Rasters:
with rasterio.open('Gaza_LC_2021-2023.tif') as pre_src:
    lc_pre = pre_src.read(1)
    nodata_pre = pre_src.nodata
with rasterio.open('Gaza_LC_2025.tif') as post_src:
    lc_post = post_src.read(1)
    nodata_post = post_src.nodata
valid_mask = (lc_pre != nodata_pre) & (lc_post != nodata_post)

#Generate DataFrames: 
df = pd.DataFrame({'LC_PreConflict': lc_pre[valid_mask].flatten(),
    'LC_PostConflict': lc_post[valid_mask].flatten()})
df = df[(df['LC_PreConflict'] != -9999) & (df['LC_PostConflict'] != -9999)]

#Reclassify:
df = df[(df['LC_PreConflict'] != 7) & (df['LC_PostConflict'] != 7)]
df = df[(df['LC_PreConflict'] != 4) & (df['LC_PostConflict'] != 4)]
label_map = {0: 'Tree Cover', 1: 'Shrubland', 2: 'Grassland', 3: 'Cropland', 5: 'Water',
    6: 'Bare', 54: 'Built-Up Low', 55: 'Built-Up High'}
plot_class_map = {k: i for i, k in enumerate(sorted(label_map.keys()))}
inv_plot_class_map = {v: k for k, v in plot_class_map.items()}  # reverse lookup for axis labels
df['LC_PreConflict_Plot'] = df['LC_PreConflict'].map(plot_class_map)
df['LC_PostConflict_Plot'] = df['LC_PostConflict'].map(plot_class_map)

#Count Transitions: 
counts = df.groupby(['LC_PreConflict_Plot', 'LC_PostConflict_Plot']).size().reset_index(name='count')
counts['size'] = np.sqrt(counts['count']) * 9

#Plot Scatter Plot: 
plt.figure(figsize=(12, 12))
plt.scatter(counts['LC_PreConflict_Plot'], counts['LC_PostConflict_Plot'], s=counts['size'],
            alpha=0.4, color='grey', edgecolors='k')
#Formatting: 
plot_ticks = sorted(inv_plot_class_map.keys())
plot_labels = [label_map[inv_plot_class_map[i]] for i in plot_ticks]
plt.xlabel('Pre-Conflict Land Cover Class', fontweight='bold', fontsize=14)
plt.ylabel('Post-Conflict Land Cover Class', fontweight='bold', fontsize=14)
plt.xticks(ticks=plot_ticks, labels=plot_labels, rotation=45)
plt.yticks(ticks=plot_ticks, labels=plot_labels)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(min(plot_ticks) - 0.5, max(plot_ticks) + 0.5)
plt.ylim(min(plot_ticks) - 0.5, max(plot_ticks) + 0.5)
#Labels: 
for _, row in counts.iterrows():
    if row['count'] > 1:
        plt.text(row['LC_PreConflict_Plot'], row['LC_PostConflict_Plot'], str(int(row['count'])),
                 ha='center', va='center', fontsize=12, color='black', fontweight='bold')
plt.tight_layout()

#%% Land Area Gridded Scatter Plot

pixel_area_km2 = 0.01  
counts['area_km2'] = counts['count'] * pixel_area_km2
counts['size'] = np.sqrt(counts['area_km2']) * 70  
#Plotting:
plt.figure(figsize=(12, 12))
plt.scatter(counts['LC_PreConflict_Plot'], counts['LC_PostConflict_Plot'], 
            s=counts['size'], alpha=0.5, color='grey', edgecolors='k')
#Formatting:
plot_ticks = sorted(inv_plot_class_map.keys())
plot_labels = [label_map[inv_plot_class_map[i]] for i in plot_ticks]
plt.xlabel('Pre-Conflict Land Cover Class', fontweight='bold', fontsize=14)
plt.ylabel('Post-Conflict Land Cover Class', fontweight='bold', fontsize=14)
plt.xticks(ticks=plot_ticks, labels=plot_labels, rotation=45)
plt.yticks(ticks=plot_ticks, labels=plot_labels)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(min(plot_ticks) - 0.5, max(plot_ticks) + 0.5)
plt.ylim(min(plot_ticks) - 0.5, max(plot_ticks) + 0.5)
#Annotation:
for _, row in counts.iterrows():
    if row['area_km2'] > 0.01:  # only label if area > 0.01 km² to avoid clutter
        plt.text(row['LC_PreConflict_Plot'], row['LC_PostConflict_Plot'], 
                 f"{row['area_km2']:.2f}", ha='center', va='center', fontsize=12, 
                 color='black', fontweight='bold')
plt.tight_layout()

#%% Bar Plot of Land Cover Percentage:

#Define LC Classes: 
lc_labels = {0: 'Tree Cover', 1: 'Shrubland', 2: 'Grassland', 3: 'Cropland', 5: 'Water',
    6: 'Bare', 54: 'Built-Up Low', 55: 'Built-Up High'}
target_classes = sorted(lc_labels.keys())

#Load Rasters: 
with rasterio.open('Gaza_LC_2021-2023.tif') as pre_src:
    lc_pre = pre_src.read(1)
    nodata_pre = pre_src.nodata
with rasterio.open('Gaza_LC_2025.tif') as post_src:
    lc_post = post_src.read(1)
    nodata_post = post_src.nodata
valid_mask = (lc_pre != nodata_pre) & (lc_post != nodata_post)
pre_vals = lc_pre[valid_mask]
post_vals = lc_post[valid_mask]

#Count Each Class: 
pre_unique, pre_counts_raw = np.unique(pre_vals, return_counts=True)
post_unique, post_counts_raw = np.unique(post_vals, return_counts=True)
pre_counts = pd.Series(pre_counts_raw, index=pre_unique).reindex(target_classes, fill_value=0)
post_counts = pd.Series(post_counts_raw, index=post_unique).reindex(target_classes, fill_value=0)

#Generate Percents: 
pre_percent = 100 * pre_counts / pre_counts.sum()
post_percent = 100 * post_counts / post_counts.sum()

#Create Bar Plot: 
x = np.arange(len(target_classes))
width = 0.45
plt.figure(figsize=(14, 8))
bars1 = plt.bar(x - width/2, pre_percent, width, label='Pre-Conflict', color='grey', edgecolor='black')
bars2 = plt.bar(x + width/2, post_percent, width, label='Post-Conflict', color='darkgrey', edgecolor='black')
#Labels:
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), 
                     textcoords="offset points", ha='center', va='bottom', fontsize=10)
autolabel(bars1)
autolabel(bars2)
#Formatting:
plt.xticks(x, [lc_labels[i] for i in target_classes], rotation=45, ha='right', fontweight='bold', fontsize=14)
plt.ylabel('Land Cover (%)', fontsize=14, fontweight='bold')
plt.legend(fontsize=14)
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()


#%% Land Cover Analysis Master Plot:

#Define Land Covers Colours: 
landcover_colors = {'Cropland': (0.941, 0.588, 1.0), 'Built-Up Low': (1.0, 0.3, 0.3),
    'Built-Up High': (0.5, 0.0, 0.0), 'Tree Cover': 'green', 'Shrubland': (1.0, 0.733, 0.133),
    'Grassland': (1.0, 1.0, 0.298), 'Bare': 'black'}
start_date = datetime(2023, 10, 7)
end_date = datetime(2025, 1, 15)
xlim_start = pd.to_datetime('2020-12-25')
xlim_end = pd.to_datetime('2025-03-01')

#Load DataFrames: 
def load_and_process(file_path, value_name, deseasonalize=True):
    df = pd.read_csv(file_path)
    date_columns = df.columns.drop('LandCover')
    clean_date_columns = [col.split('.')[0] for col in date_columns]
    dates = pd.to_datetime(clean_date_columns, format='%Y_%m_%d')
    data = df.set_index('LandCover')
    data.columns = dates
    data = data[~data.index.isin(['Water'])]
    data_long = data.T.reset_index()
    data_long = pd.melt(data_long, id_vars='index', var_name='LandCover', value_name=value_name)
    data_long.rename(columns={'index': 'Date'}, inplace=True)
    data_long['Month'] = data_long['Date'].dt.month
    if deseasonalize:
        monthly_means = data_long.groupby(['LandCover', 'Month'])[value_name].mean().reset_index()
        monthly_means.rename(columns={value_name: 'LongTerm_MonthlyMean'}, inplace=True)
        data_long = data_long.merge(monthly_means, on=['LandCover', 'Month'], how='left')
        data_long[f'{value_name}_Deseasonalized'] = data_long[value_name] - data_long['LongTerm_MonthlyMean']
        value_col = f'{value_name}_Deseasonalized'
    else:
        value_col = value_name
    data_long['MonthStart'] = data_long['Date'].dt.to_period('M').dt.to_timestamp()
    monthly_agg = data_long.groupby(['MonthStart', 'LandCover'])[value_col].mean().reset_index()
    plot_data = monthly_agg.pivot(index='MonthStart', columns='LandCover', values=value_col)
    return plot_data

#Plot Time Seires: 
def plot_panel(ax, plot_data, ylabel, title, legend_position=None):
    for col in plot_data.columns:
        color = landcover_colors.get(col, (0.5, 0.5, 0.5))
        ax.plot(plot_data.index, plot_data[col], label=col, color=color)
    ax.axhline(y=0, color='k')
    ax.axvline(start_date, color='grey', linewidth=0.5)
    ax.axvline(end_date, color='grey', linewidth=0.5)
    ax.axvspan(start_date, end_date, color='grey', alpha=0.25)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True)
    if legend_position:
        ax.legend(frameon=True, fontsize=8, loc=legend_position)
#Mean Plot:
def plot_mean_std(ax, deseason_data, ylabel, title, legend_position=None):
    mean_vals = deseason_data.mean(axis=1)
    std_vals = deseason_data.std(axis=1)
    ax.plot(mean_vals.index, mean_vals, color='black', label='Mean Deseasonalized')
    ax.errorbar(mean_vals.index, mean_vals, yerr=std_vals, fmt='none',
                ecolor='black', alpha=0.75, capsize=4, label='±1 Std Dev')
    ax.axhline(y=0, color='k')
    ax.axvline(start_date, color='grey', linewidth=0.5)
    ax.axvline(end_date, color='grey', linewidth=0.5)
    ax.axvspan(start_date, end_date, color='grey', alpha=0.25)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True)
    if legend_position:
        ax.legend(frameon=True, fontsize=8, loc=legend_position)

#Load Data:
ndvi_raw = load_and_process('Gaza_NDVI_AnalysisVals.csv', 'NDVI', deseasonalize=False)
ndvi_deseason = load_and_process('Gaza_NDVI_AnalysisVals.csv', 'NDVI', deseasonalize=True)
ndbi_raw = load_and_process('Gaza_NDBI_AnalysisVals.csv', 'NDBI', deseasonalize=False)
ndbi_deseason = load_and_process('Gaza_NDBI_AnalysisVals.csv', 'NDBI', deseasonalize=True)

#Establish Subplots:
fig, axs = plt.subplots(3, 2, figsize=(20, 13), sharex=True)
#Row 1, Raw:
plot_panel(axs[0,0], ndvi_raw, 'Raw NDVI Value', '', legend_position='upper right')
plot_panel(axs[0,1], ndbi_raw, 'Raw NDBI Value', '')
# Row 2, Deseasonalized
plot_panel(axs[1,0], ndvi_deseason, 'Deseasonalized NDVI Value', '')
plot_panel(axs[1,1], ndbi_deseason, 'Deseasonalized NDBI Value', '')
# Row 3, Mean + Std Dev:
plot_mean_std(axs[2,0], ndvi_deseason, 'Mean Deseasonalized NDVI Value', '', 
              legend_position='lower left')
plot_mean_std(axs[2,1], ndbi_deseason, 'Mean Deseasonalized NDBI Value', '')
#Formatting: 
for ax in axs[2,:]:
    ax.set_xlabel('Date', fontweight='bold', fontsize=13)
    ax.set_xlim(xlim_start, xlim_end)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
plt.tight_layout()

#%% ANOVA Analysis: 
    
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import f_oneway
import numpy as np

#Manipulate Data: 
def load_ndvi_or_ndbi(file_path, value_label):
    df = pd.read_csv(file_path)
    date_columns = df.columns.drop('LandCover')
    clean_date_columns = [col.split('.')[0] for col in date_columns]
    dates = pd.to_datetime(clean_date_columns, format='%Y_%m_%d')
    data = df.set_index('LandCover')
    data.columns = dates
    data = data[~data.index.isin(['Water'])]
    data_long = data.T.reset_index()
    data_long = pd.melt(data_long, id_vars='index', var_name='LandCover', value_name=value_label)
    data_long.rename(columns={'index': 'Date'}, inplace=True)
    conflict_start = pd.to_datetime('2023-10-07')
    data_long['ConflictPeriod'] = np.where(data_long['Date'] < conflict_start, 'Pre-Conflict', 'Conflict')
    return data_long
def prepare_violin_data(data_long, value_col):
    pre_vals = data_long[data_long['ConflictPeriod'] == 'Pre-Conflict'][value_col].dropna()
    conflict_vals = data_long[data_long['ConflictPeriod'] == 'Conflict'][value_col].dropna()
    
    #ANOVA Test: 
    f_stat, p_val = f_oneway(pre_vals, conflict_vals)
    #Effect size (η² = SS_between / SS_total):
    group_means = [np.mean(pre_vals), np.mean(conflict_vals)]
    overall_mean = np.mean(np.concatenate([pre_vals, conflict_vals]))
    ss_between = sum(len(g) * (m - overall_mean) ** 2 for g, m in zip([pre_vals, conflict_vals], group_means))
    ss_total = sum((x - overall_mean) ** 2 for x in np.concatenate([pre_vals, conflict_vals]))
    eta_sq = ss_between / ss_total if ss_total > 0 else 0
    return [pre_vals.values, conflict_vals.values], ['Pre-Conflict', 'Conflict'], f_stat, p_val, eta_sq

#Load Data:
ndvi_long = load_ndvi_or_ndbi('Gaza_NDVI_AnalysisVals.csv', 'NDVI')
ndbi_long = load_ndvi_or_ndbi('Gaza_NDBI_AnalysisVals.csv', 'NDBI')
ndvi_data, ndvi_labels, ndvi_f, ndvi_p, ndvi_eta = prepare_violin_data(ndvi_long, 'NDVI')
ndbi_data, ndbi_labels, ndbi_f, ndbi_p, ndbi_eta = prepare_violin_data(ndbi_long, 'NDBI')

#Plot Figures: 
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for ax, data, labels, f_val, p_val, eta_sq, varname in zip(axs, [ndvi_data, ndbi_data], [ndvi_labels, ndbi_labels],
    [ndvi_f, ndbi_f], [ndvi_p, ndbi_p], [ndvi_eta, ndbi_eta], ['NDVI', 'NDBI']):
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('darkgrey')
        pc.set_edgecolor('black')
        pc.set_alpha(0.65)
    for i, group in enumerate(data):
        mean = np.mean(group)
        median = np.median(group)
        min_val = np.min(group)
        max_val = np.max(group)
        ax.hlines(mean, i + 0.8, i + 1.2, color='black', linestyle='-', linewidth=2,
                  label='Mean' if i == 0 and varname == 'NDVI' else "")
        ax.hlines(median, i + 0.8, i + 1.2, color='black', linestyle='--', linewidth=2,
                  label='Median' if i == 0 and varname == 'NDVI' else "")
        ax.hlines([min_val, max_val], i + 0.9, i + 1.1, color='black', linewidth=1)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, fontweight='bold', fontsize=12)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.grid(True, axis='y')
    sci_p = f"{p_val:.2E}"
    stat_text = f"ANOVA F = {f_val:.2f}\nP = {sci_p}\nη² = {eta_sq:.3f}"
    ax.text(0.5, 0.95, stat_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='center',
            fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))
    axs[1].yaxis.set_tick_params(labelleft=True)
#Legend:
axs[0].set_ylabel('NDVI Value', fontweight='bold', fontsize=12)
axs[1].set_ylabel('NDBI Value', fontweight='bold', fontsize=12)
axs[0].legend(loc='lower left', frameon=True, fontsize=12)
plt.tight_layout()
plt.show()

#%% Per Land Cover Class ANOVA Plot: 
    
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import f_oneway
import numpy as np
import matplotlib.gridspec as gridspec

# Load CSVs
ndvi_df = pd.read_csv('Gaza_NDVI_AnalysisVals.csv')
ndbi_df = pd.read_csv('Gaza_NDBI_AnalysisVals.csv')

# Prepare long-format data
def prepare_data(df, value_label):
    date_columns = df.columns.drop('LandCover')
    clean_date_columns = [col.split('.')[0] for col in date_columns]
    dates = pd.to_datetime(clean_date_columns, format='%Y_%m_%d')
    df = df.set_index('LandCover')
    df.columns = dates
    df = df[~df.index.isin(['Water'])]
    df_long = df.T.reset_index().melt(id_vars='index', var_name='LandCover', value_name=value_label)
    df_long.rename(columns={'index': 'Date'}, inplace=True)
    conflict_start = pd.to_datetime('2023-10-07')
    df_long['ConflictPeriod'] = np.where(df_long['Date'] < conflict_start, 'Pre-Conflict', 'Conflict')
    return df_long

ndvi_long = prepare_data(ndvi_df, 'NDVI')
ndbi_long = prepare_data(ndbi_df, 'NDBI')

# Get land cover classes
landcover_classes = sorted(ndvi_long['LandCover'].unique())
nrows = len(landcover_classes) + 1  # include overall
ncols = 2

# Create figure
fig = plt.figure(figsize=(14, 4 * nrows))
gs = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.25, hspace=0.6)
summary_stats = []

# Violin plotting function
def plot_violin(ax, data, show_ylabel=False, show_legend=False, ylabel_text=None, title_text=None):
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('grey')
        pc.set_edgecolor('black')
        pc.set_alpha(0.75)

    for i, group in enumerate(data):
        mean = np.mean(group)
        median = np.median(group)
        min_val = np.min(group)
        max_val = np.max(group)
        ax.hlines(mean, i + 0.8, i + 1.2, color='black', linestyle='-', linewidth=2,
                  label='Mean' if i == 0 and show_legend else "")
        ax.hlines(median, i + 0.8, i + 1.2, color='dimgray', linestyle='--', linewidth=2,
                  label='Median' if i == 0 and show_legend else "")
        ax.hlines([min_val, max_val], i + 0.9, i + 1.1, color='black', linewidth=1)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Pre-Conflict', 'Conflict'], fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=1)
    ax.grid(True, axis='y', linestyle=':', linewidth=0.5, color='lightgrey')
    if show_ylabel:
        ax.set_ylabel(ylabel_text, fontweight='bold')
    if show_legend:
        ax.legend(loc='upper right', frameon=False, fontsize=10)
    if title_text:
        ax.set_title(title_text, fontsize=12, fontweight='bold')

# Process and plot for each class
def process_and_plot(ax, df, varname, lc=None, show_ylabel=False, show_legend=False):
    if lc:
        df = df[df['LandCover'] == lc]

    pre_vals = df[df['ConflictPeriod'] == 'Pre-Conflict'][varname].dropna()
    conflict_vals = df[df['ConflictPeriod'] == 'Conflict'][varname].dropna()

    f_stat, p_val = f_oneway(pre_vals, conflict_vals)
    summary_stats.append({
        'LandCover': lc if lc else 'Overall',
        'Variable': varname,
        'Mean_Pre': np.mean(pre_vals),
        'Median_Pre': np.median(pre_vals),
        'Mean_Conflict': np.mean(conflict_vals),
        'Median_Conflict': np.median(conflict_vals),
        'F_stat': f_stat,
        'p_val': p_val,
        'Significant': p_val < 0.05
    })

    title = lc if lc else "Overall"
    plot_violin(ax, [pre_vals, conflict_vals],
                show_ylabel=show_ylabel,
                show_legend=show_legend,
                ylabel_text=f"{varname}",
                title_text=title)
    stat_text = f"F = {f_stat:.2f}\nP = {p_val:.2E}"
    ax.text(0.5, 0.95, stat_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='center',
            fontsize=10, bbox=dict(facecolor='white', edgecolor='black', alpha=0.85))
    ax.grid(True)

# Loop through land cover classes
for i, lc in enumerate(landcover_classes):
    ax_ndvi = fig.add_subplot(gs[i, 0])
    ax_ndbi = fig.add_subplot(gs[i, 1])
    process_and_plot(ax_ndvi, ndvi_long, 'NDVI', lc, show_ylabel=True, show_legend=(i == 0))
    process_and_plot(ax_ndbi, ndbi_long, 'NDBI', lc, show_ylabel=True)

# Final row for overall
ax_ndvi = fig.add_subplot(gs[-1, 0])
ax_ndbi = fig.add_subplot(gs[-1, 1])
process_and_plot(ax_ndvi, ndvi_long, 'NDVI', lc=None, show_ylabel=True)
process_and_plot(ax_ndbi, ndbi_long, 'NDBI', lc=None, show_ylabel=True)

# Save and show
plt.tight_layout(rect=(0, 0, 1, 1))  # no space needed on left anymore
fig.savefig("NDVI_NDBI_ANOVA_ViolinPlots_Grayscale_WithTitles.png", dpi=300)
pd.DataFrame(summary_stats).to_csv("ANOVA_Summary_With_Significance.csv", index=False)

#%% Mean Difference Plot: 
    
#Load Data and Reclassify Water Class: 
df = pd.read_csv('Gaza_SamplePoints_AnalysisVals.csv')

# Create Scatter Plot Function
def ScatterPlot(df1, df2, colour, colour2, Label1, Label2):
    # Line of Best Fit
    coefficients = np.polyfit(df1, df2, 1)
    line_of_best_fit = np.poly1d(coefficients)
    # Generate Scatter Plot  
    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax1.scatter(df1, df2, color=colour, s=15, alpha=0.75)
    ax1.plot([-1, 1], [-1, 1], color='black', linestyle='--', linewidth=1)
    ax1.plot(df1, line_of_best_fit(df1), color=colour2)
    # Plot Formatting
    ax1.set_xlim(-0.35, 0.35)
    ax1.set_ylim(-0.35, 0.35)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.75)
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.75)
    ax1.set_xlabel(Label1, fontweight='bold', fontsize=14)
    ax1.set_ylabel(Label2, fontweight='bold', fontsize=14)
    ax1.grid(True)
    # Add quadrant labels
    ax1.text(0.2, 0.25, '+ΔNDVI +ΔNDBI\nStagnant Water \n Algal Blooms \n Agriculture in Refugee Camps', fontsize=13, ha='center', va='center',
             color='grey', weight='bold')
    ax1.text(-0.2, 0.25, '–ΔNDVI +ΔNDBI\nRubble Development, \nRefugee Camps \nMilitary Infrastructure', fontsize=13, ha='center', va='center',
             color='grey', weight='bold')
    ax1.text(-0.2, -0.25, '–ΔNDVI –ΔNDBI\nComplete Destruction of \n Natural & Built Environments', fontsize=13, ha='center', va='center',
             color='grey', weight='bold')
    ax1.text(0.2, -0.25, '+ΔNDVI –ΔNDBI\nAbandoned \n or \nUnregulated Agriculture', fontsize=13, ha='center', va='center',
             color='grey', weight='bold')
    # Spearman's Rho
    correlation, p_value = spearmanr(df1, df2)
    print('Spearman Correlation Coefficient:', correlation)
    print('p-Value:', p_value)
    correlation = str(correlation)[:5]
    p_value = Decimal(p_value)
    p_value = "{:.1E}".format(p_value)
    ax1.text(0.25, 0.075, f'Rho: {correlation}\np-Value: {p_value}',
             fontsize=15, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8))

#Generate Scatterplot: 
ScatterPlot(df['Mean_NDVI_Difference'], df['Mean_NDBI_Difference'], 'grey', 'black', 
            'NDVI Mean Difference', 'NDBI Mean Difference')

#%% End of Code