# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 09:48:40 2020

@author: bergs
"""

import pandas as pd
import matplotlib.pyplot as plt

# import os
# import numpy as np
# from scipy import stats
# import urllib.request
# import math
# import statistics 
# import seaborn as sns

file_name = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
DataFrame = pd.read_csv(file_name)

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

DataFrame.tamponi/DataFrame.totale_casi
DataFrame.deceduti/DataFrame.totale_casi

# Tamponi / Totale Attualmente Positivi
x = DataFrame.data
y = DataFrame.tamponi/DataFrame.totale_attualmente_positivi
x_label = "Data"
y_label = "Tamponi:Tot. attualmente positivi"
figname = str(y_label+".jpg")
title = y_label
plot(x, y, title, x_label, y_label, plot_x_size, plot_y_size, figname)