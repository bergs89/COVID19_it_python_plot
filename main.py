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

def plot(x, y, title, x_label, y_label, plot_x_size, plot_y_size, legend, figname, y_max=None, y_min=None):
    plt.figure(figsize=(plot_x_size, plot_y_size))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45)
    plt.grid()
    plt.plot(x, y, marker='o', label=str(legend))
    plt.legend(loc="upper left", ncol=2, title="Legend", fancybox=True)
    plt.ylim(ymax=y_max, ymin=y_min)
    plt.savefig(figname)
    return

def bar(x, y, title, x_label, y_label, plot_x_size, plot_y_size, legend, figname, y_max=None, y_min=None):
    plt.figure(figsize=(plot_x_size, plot_y_size))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45)
    plt.grid()
    plt.bar(x, y, label=str(legend))
    plt.legend(loc="upper left", ncol=2, title="Legend", fancybox=True)
    plt.ylim(ymax=y_max, ymin=y_min)
    plt.savefig(figname)
    return

# Plot nazionali
for column in DataFrame:
    if column == "data":
        continue
    elif column == "stato":
        continue
    elif column == "note_it":
        continue
    elif column == "note_en":
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
        if column == "nuovi_attualmente_positivi":
            bar(x, y, title, x_label, y_label, plot_x_size, plot_y_size, legend, figname)
        else:
            plot(x, y, title, x_label, y_label, plot_x_size, plot_y_size, legend, figname)
        continue

# PLOTS
# Picture setup<
plot_x_size = 16
plot_y_size = 10

# 1_Tasso di decessi percentuale
# 2_Pazienti positivi in piu al giorno
# 3_Tamponi / Totale Attualmente Positivi
# 4_Guariti diviso nuovi attualmente positivi

x = [DataFrame.data, DataFrame.data, DataFrame.data, DataFrame.data]
y = [DataFrame.deceduti/DataFrame.totale_casi*100, DataFrame.nuovi_attualmente_positivi, DataFrame.tamponi/DataFrame.totale_attualmente_positivi, DataFrame.dimessi_guariti/DataFrame.nuovi_attualmente_positivi]
x_label = ["Data", "Data", "Data", "Data"]
y_label = ["Tasso decessi percentuale", "Casi positivi giornalieri", "Tamponi diviso Tot. attualmente positivi", "Guariti diviso nuovi attualmente positivi"]
title = ["Tasso decessi percentuale", "Casi positivi giornalieri", "Tamponi diviso Tot. attualmente positivi", "Guariti diviso nuovi attualmente positivi"]
legend = ["Tasso decessi percentuale", "Casi positivi giornalieri", "Tamponi diviso Tot. attualmente positivi", "Guariti diviso nuovi attualmente positivi"]
figname = [str(title[0]+".jpg"), str(title[1]+".jpg"), str(title[2]+".jpg"), str(title[3]+".jpg")]
for i in range(0,len(x)):
    plot(x[i], y[i], title[i], x_label[i], y_label[i], plot_x_size, plot_y_size, legend[i], figname[i])

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

#Tasso di crescita giornaliero 
x = dates
y = ratio_positivi
title = "Tasso di crescita giornaliero"
x_label = "Data"
y_label = "Tasso di crescita giornaliero: positivi al giorno n+1 / positivi al giorno n"
legend = ratio_positivi.columns.values
figname = "Tasso di crescita giornaliero.jpg"
y_max = 3
y_min = 0.5
plot(x, y, title, x_label, y_label, plot_x_size, plot_y_size, legend, figname, y_max, y_min)

#Tasso di crescita giornaliero w/ expansion
ma_days=2
x = dates
y = moving_average = ratio_positivi.expanding(min_periods=ma_days).mean()
title = "Tasso di crescita giornaliero using expansion"
x_label = "Data"
y_label = "Tasso di crescita giornaliero: positivi al giorno n+1 / positivi al giorno n, with expansion"
legend = moving_average.columns.values
figname = "Tasso di crescita giornaliero with expansion.jpg"
y_max = 3
plot(x, y, title, x_label, y_label, plot_x_size, plot_y_size, legend, figname, y_max)

# Tasso di crescita giornaliero per regioni
ma_days=1
# Tasso di crescita giornaliero per regioni w/expansion
for i in range(1,7):
    date=dates[len(dates)-i]
    # TODO media mobile non mi piace, falsa molto la realta
    y = ratio_positivi[:].iloc[len(dates)-i]
    mean = DataFrame.totale_casi.shift(0)[:].iloc[-i]/DataFrame.totale_casi.shift(1)[:].iloc[-i]
    plt.figure(figsize=(plot_x_size*1.5, plot_y_size))
    plt.title("Ultimo tasso di crescita giornaliero al "+date)
    plt.xlabel("Regione")
    plt.ylabel("Coefficiente giornaliero")
    plt.xticks(rotation=45)
    y_pos=np.arange(len(regions))
    plt.bar(regions, y)
    plt.hlines(mean, linestyle='dashed', colors='red', label='Average', xmin='Abruzzo', xmax='Veneto')
    plt.hlines(1, linestyle='dashed', colors='black', label='Target', xmin='Abruzzo', xmax='Veneto')
    plt.legend(loc="upper left", title="Legend", fancybox=True)
    plt.ylim(ymax=1.75)
    date=date.replace(":",".")
    plt.savefig("Ultimo tasso di crescita giornaliero al "+date+" .jpg")

images = []
files = [f for f in os.listdir('.') if os.path.isfile(f)]
for file in files:
    if file.startswith("Ultimo tasso di crescita giornaliero") & file.endswith(" .jpg"):
        images.append(imageio.imread(file))
imageio.mimsave('./Tassi di crescita giornalieri.gif', images, duration = 1)


def analisi_CNS(DataFrame, DataFrame_regions):
    # Analisi Centro Nord Sud
    # Estarzione dati
    df=DataFrame_regions
    df_nord=pd.DataFrame()
    df_centro=pd.DataFrame()
    df_sud=pd.DataFrame()

    for region in regions:
        if region in ["Abruzzo", "Toscana", "Marche", "Umbria", "Lazio", "Molise"]:
            df_centro[region] = df[df['denominazione_regione'].str.contains(str(region))].totale_casi.reset_index().totale_casi
        if region in ["Basilicata", "Campania", "Puglia", "Calabria", "Sicilia", "Sardegna"]:      
            df_sud[region]=df_centro[region] = df[df['denominazione_regione'].str.contains(str(region))].totale_casi.reset_index().totale_casi
        if region in ["Liguria", "Piemonte", "Valle d'Aosta", "Lombardia", "P.A. Trento", "Friuli Venezia Giulia", "Veneto", "P.A. Bolzano", "Emilia Romagna"]:
            df_nord[region]=df_centro[region] = df[df['denominazione_regione'].str.contains(str(region))].totale_casi.reset_index().totale_casi
 
    centro_totale_positivi=pd.DataFrame()
    centro_totale_positivi=df_centro.sum(axis=1)-df_nord.sum(axis=1)-df_sud.sum(axis=1)
    sud_totale_positivi=df_sud.sum(axis=1)
    nord_totale_positivi=df_nord.sum(axis=1)
        
    for region in regions:
        if region in ["Abruzzo", "Toscana", "Marche", "Umbria", "Lazio", "Molise"]:
            df_centro[region] = df[df['denominazione_regione'].str.contains(str(region))].nuovi_attualmente_positivi.reset_index().nuovi_attualmente_positivi
        if region in ["Basilicata", "Campania", "Puglia", "Calabria", "Sicilia", "Sardegna"]:      
            df_sud[region]=df_centro[region] = df[df['denominazione_regione'].str.contains(str(region))].nuovi_attualmente_positivi.reset_index().nuovi_attualmente_positivi
        if region in ["Liguria", "Piemonte", "Valle d'Aosta", "Lombardia", "P.A. Trento", "Friuli Venezia Giulia", "Veneto", "P.A. Bolzano", "Emilia Romagna"]:
            df_nord[region]=df_centro[region] = df[df['denominazione_regione'].str.contains(str(region))].nuovi_attualmente_positivi.reset_index().nuovi_attualmente_positivi
 
    centro_nuovi_positivi=pd.DataFrame()
    centro_nuovi_positivi=df_centro.sum(axis=1)-df_nord.sum(axis=1)-df_sud.sum(axis=1)
    sud_nuovi_positivi=df_sud.sum(axis=1)
    nord_nuovi_positivi=df_nord.sum(axis=1)
    return nord_totale_positivi, centro_totale_positivi, sud_totale_positivi, nord_nuovi_positivi, centro_nuovi_positivi, sud_nuovi_positivi

nord_totale_positivi, centro_totale_positivi, sud_totale_positivi, nord_nuovi_positivi, centro_nuovi_positivi, sud_nuovi_positivi = analisi_CNS(DataFrame, DataFrame_regions)
# Plot analisi Italia Nord Centro Sud
x = [dates, dates, dates, dates, dates, dates]
y = [nord_totale_positivi, centro_totale_positivi, sud_totale_positivi, nord_nuovi_positivi, centro_nuovi_positivi, sud_nuovi_positivi]
x_label = ["Data", "Data", "Data", "Data",  "Data", "Data"]
y_label = ["Persone", "Persone", "Persone", "Persone", "Persone", "Persone"]
title = ["Totale positivi al Nord", "Totale positivi al Centro", "Totale positivi al Sud", "Nuovi positivi al Nord", "Nuovi positivi al Centro",  "Nuovi positivi al Sud", ]
legend = ["Totale positivi al Nord", "Totale positivi al Centro", "Totale positivi al Sud", "Nuovi positivi al Nord", "Nuovi positivi al Centro",  "Nuovi positivi al Sud", ]
figname = [str(title[0]+".jpg"), str(title[1]+".jpg"), str(title[2]+".jpg"), str(title[3]+".jpg"),  str(title[4]+".jpg"),  str(title[5]+".jpg")]
for i in range(0,len(x)):
    bar(x[i], y[i], title[i], x_label[i], y_label[i], plot_x_size, plot_y_size, legend[i], figname[i])

