# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 09:48:40 2020

@author: bergs
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os

# from scipy import stats

file_name_it = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
file_name_regions = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv"
DataFrame = pd.read_csv(file_name_it)
DataFrame_regions = pd.read_csv(file_name_regions)

def plot(x, y, title, x_label, y_label, plot_x_size, plot_y_size, legend, figname, y_max=None):
    plt.figure(figsize=(plot_x_size, plot_y_size))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45)
    plt.grid()
    plt.plot(x, y, marker='o', label=str(legend))
    plt.legend(legend, loc="upper left", ncol=2, title="Legend", fancybox=True)
    plt.ylim(ymax=y_max)
    plt.savefig(figname)
    plt.show()
    return

#Analisi nazionali
    
#Analisi regionali


# Plot nazionali
for column in DataFrame:
    if column == "data":
        continue
    elif column == "stato":
        continue
    else:
        plot_x_size = 16
        plot_y_size = 10
        x = DataFrame.data
        y = DataFrame[column]
        x_label = "Data"
        y_label = "Persone"
        legend = column
        figname = str(column+".jpg")
        title = str(column).replace("_"," ")
        plot(x, y, title, x_label, y_label, plot_x_size, plot_y_size, legend, figname)
        continue

# Tasso di decessi percentuale
x = DataFrame.data
y = tasso_decessi = DataFrame.deceduti/DataFrame.totale_casi*100
x_label = "Data"
y_label = title = legend = "Tasso decessi percentuale"
figname = str(title+".jpg")
plot(x, y, title, x_label, y_label, plot_x_size, plot_y_size, legend, figname)

#Pazienti positivi in piu al giorno
DataFrame['d_totale_casi'] = DataFrame["totale_casi"].diff(1)
x = DataFrame.data
y = DataFrame.d_totale_casi
legend = title = y_label = "Casi positivi giornalieri"
x_label = "Data"
y_label = "Persone"
figname = "Casi positivi giornalieri.jpg"
plot(x, y, title, x_label, y_label, plot_x_size, plot_y_size, legend, figname)

# Tamponi / Totale Attualmente Positivi
x = DataFrame.data
y = DataFrame.tamponi/DataFrame.totale_attualmente_positivi
x_label = "Data"
y_label = title = legend = "Tamponi_Tot. attualmente positivi"
figname = str(y_label+".jpg")
plot(x, y, title, x_label, y_label, plot_x_size, plot_y_size, legend, figname)

#fare i plot in ciclo for per tutte le liste come si deve

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

#Tasso di crescita giornaliero MA
ma_days=3
x = dates
y = moving_average = ratio_positivi.expanding(min_periods=ma_days).mean()
title = "Tasso di crescita giornaliero MA "+str(ma_days)+" days"
x_label = "Data"
y_label = "Tasso di crescita giornaliero: positivi al giorno n+1 / positivi al giorno n"
legend = moving_average.columns.values
figname = "Tasso di crescita giornaliero MA "+str(ma_days)+" days.jpg"
y_max = 2
plot(x, y, title, x_label, y_label, plot_x_size, plot_y_size, legend, figname, y_max)

# Tasso di crescita giornaliero per regioni
ma_days=1
for i in range(1,7):
    date=dates[len(dates)-i]
    last_moving_average_crescita_giornaliera=moving_average[:].iloc[len(dates)-i]
    mean = DataFrame.totale_casi.shift(0)[:].iloc[-i]/DataFrame.totale_casi.shift(1)[:].iloc[-i]
    plt.figure(figsize=(plot_x_size*1.5, plot_y_size))
    plt.title("Ultimo tasso di crescita giornaliero al "+date)
    plt.xlabel("Regione")
    plt.ylabel("Coefficiente giornaliero")
    plt.xticks(rotation=45)
    y_pos=np.arange(len(regions))
    plt.bar(regions, last_moving_average_crescita_giornaliera.T.reset_index().iloc[:,1])
    plt.hlines(mean, linestyle='dashed', colors='red', label='Average', xmin='Abruzzo', xmax='Veneto')
    plt.hlines(1, linestyle='dashed', colors='black', label='Target', xmin='Abruzzo', xmax='Veneto')
    plt.legend(loc="upper left", title="Legend", fancybox=True)
    plt.ylim(ymax=1.75)
    date=date.replace(":",".")
    plt.savefig("Ultimo tasso di crescita giornaliero al "+date+".jpg")

# Tasso di crescita regionali GIF
images = []
files = [f for f in os.listdir('.') if os.path.isfile(f)]
for file in files:
    if file.startswith("Ultimo tasso di crescita giornaliero") & file.endswith(".jpg"):
        images.append(imageio.imread(file))
imageio.mimsave('./Tassi di crescita giornalieri.gif', images, duration = 0.5)