# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 09:48:40 2020

@author: bergs
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
import imageio
import os

# from scipy import stats
# import urllib.request
# import math
# import seaborn as sns

file_name_it = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
file_name_regions = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv"
DataFrame = pd.read_csv(file_name_it)
DataFrame_regions = pd.read_csv(file_name_regions)

def plot(x, y, title, x_label, y_label, plot_x_size, plot_y_size, figname):
    plt.figure(figsize=(plot_x_size, plot_y_size))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45)
    plt.grid()
    plt.plot(x, y, marker='o', label=str(y_label))
    plt.legend(loc="upper left", ncol=2, title="Legend", fancybox=True)
    plt.savefig(figname)
    plt.show()
    return

for column in DataFrame:
    if column == "data":
        continue
    elif column == "stato":
        continue
    else:
        x = DataFrame.data
        y = DataFrame[column]
        x_label = "Data"
        y_label = column
        plot_x_size = 16
        plot_y_size = 10
        figname = str(column+".jpg")
        title = str(column).replace("_"," ")
        plot(x, y, title, x_label, y_label, plot_x_size, plot_y_size, figname)
        continue

# Tasso di decessi percentuale
x = DataFrame.data
y = tasso_decessi = DataFrame.deceduti/DataFrame.totale_casi*100
x_label = "Data"
y_label = "Tasso decessi percentuale"
figname = str(y_label+".jpg")
title = y_label
plot(x, y, title, x_label, y_label, plot_x_size, plot_y_size, figname)

#Pazienti positivi in piu al giorno
DataFrame['d_totale_casi'] = DataFrame["totale_casi"].diff(1)
z = DataFrame.data
y = DataFrame.d_totale_casi
plt.figure(figsize=(plot_x_size, plot_y_size))
plt.title("Casi positivi giornalieri")
plt.xlabel("Data")
plt.ylabel("Persone")
plt.bar(x, y, label="Casi positivi giornalieri")
plt.legend(loc="upper left", title="Legend", fancybox=True)
plt.xticks(rotation=45)
plt.savefig("Casi positivi giornalieri.jpg")
plt.show()

# Tamponi / Totale Attualmente Positivi
x = DataFrame.data
y = DataFrame.tamponi/DataFrame.totale_attualmente_positivi
x_label = "Data"
y_label = "Tamponi diviso Tot. attualmente positivi"
figname = str(y_label+".jpg")
title = y_label
plot(x, y, title, x_label, y_label, plot_x_size, plot_y_size, figname)

# Regional Analysis
# Costruzione delle variabili
regions=DataFrame_regions.denominazione_regione.unique()
x=DataFrame_regions.denominazione_regione
y=DataFrame_regions.totale_casi
dates=DataFrame_regions.data.unique()
regions=DataFrame_regions.denominazione_regione.unique()
positivi=pd.DataFrame()
deceduti=pd.DataFrame()
dates_num = pd.Series(range(1,len(dates)+1))
for region in regions:
    column_tmp = DataFrame_regions.loc[DataFrame_regions['denominazione_regione'] == region, 'totale_casi']
    column_tmp = column_tmp.reset_index()
    positivi[region] = column_tmp.totale_casi
    rate_positivi = positivi.diff(axis = 0, periods = 1) 
    ratio_positivi = positivi.div(positivi.shift(1)).fillna(value=1)
    ratio_positivi = ratio_positivi.replace(np.inf, 1)
    column_tmp2 = DataFrame_regions.loc[DataFrame_regions['denominazione_regione'] == region, 'deceduti']
    column_tmp2 = column_tmp2.reset_index()
    deceduti[region] = column_tmp2.deceduti
       
# Picture setup
plot_x_size = 16
plot_y_size = 10

#Variazione giornaliera positivi
plt.figure(figsize=(plot_x_size, plot_y_size))
plt.title("Variazione gironaliera dei positivi")
plt.xlabel("Data")
plt.ylabel("Persone")
plt.xticks(rotation=45)
plt.grid()
for region in regions:
    plt.plot(dates, rate_positivi[region], marker='o',  label=str(region))
    plt.legend(loc="upper left", ncol=2, title="Legend", fancybox=True)
plt.savefig("Variazione gironaliera dei positivi.jpg")
plt.show()

#Tasso di crescita giornaliero MA
ma_days=3
y = moving_average = ratio_positivi.expanding(min_periods=ma_days).mean()
plt.figure(figsize=(plot_x_size, plot_y_size))
plt.title("Tasso di crescita giornaliero MA "+str(ma_days)+" days")
plt.xlabel("Data")
plt.ylabel("Persone")
plt.xticks(rotation=45)
plt.grid()
plt.legend(regions)
plt.plot(dates, moving_average, marker='o')
plt.ylim(ymax=2.5)
plt.legend(moving_average.columns.values, loc="upper left", ncol=2, title="Legend", fancybox=True)
plt.savefig("Tasso di crescita giornaliero MA "+str(ma_days)+" days.jpg")
plt.show()

# Tasso di crescita giornaliero per regioni
ma_days=1
for i in range(1,7):
    date=dates[len(dates)-i]
    last_moving_average_crescita_giornaliera=moving_average[:].iloc[len(dates)-i]
    # TODO wrong mean
    #mean=statistics.mean(last_moving_average_crescita_giornaliera)
    #mean = DataFrame.totale_casi[:].iloc[-(len(dates)-i-1)]/DataFrame.totale_casi[:].iloc[-(len(dates)-i)]
    plt.figure(figsize=(plot_x_size*1.5, plot_y_size))
    plt.title("Ultimo tasso di crescita giornaliero al "+date)
    plt.xlabel("Regione")
    plt.ylabel("Coefficiente giornaliero")
    plt.xticks(rotation=45)
    y_pos=np.arange(len(regions))
    plt.bar(regions, last_moving_average_crescita_giornaliera.T.reset_index().iloc[:,1])
    #plt.hlines(mean, linestyle='dashed', colors='red', label='Average', xmin='Abruzzo', xmax='Veneto')
    #plt.hlines(1, linestyle='dashed', colors='black', label='Target', xmin='Abruzzo', xmax='Veneto')
    plt.legend(loc="upper left", title="Legend", fancybox=True)
    plt.ylim(ymax=1.75)
    date=date.replace(":",".")
    plt.savefig("Ultimo tasso di crescita giornaliero al "+date+".jpg")
    plt.show()

# Tasso di crescita regionali GIF
images = []
files = [f for f in os.listdir('.') if os.path.isfile(f)]
for file in files:
    if file.startswith("Ultimo tasso di crescita giornaliero") & file.endswith(".jpg"):
        images.append(imageio.imread(file))
imageio.mimsave('./Tassi di crescita giornalieri.gif', images, duration = 0.5)