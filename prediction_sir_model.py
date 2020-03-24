# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 12:26:50 2020

@author: bergs
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
from scipy.integrate import odeint
import matplotlib.pyplot as plt

file_name_it = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
file_name_regions = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv"
DataFrame = pd.read_csv(file_name_it)
DataFrame_regions = pd.read_csv(file_name_regions)

def SIR_model(I0, R0, N, beta, gamma, t_max=len(DataFrame.data)):
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0
    # A grid of time points (in days)
    t = np.linspace(0, t_max, t_max)
    # The SIR model differential equations.
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    # Initial conditions vector
    y0 = S0, I0, R0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    return S, I, R, t

def calibration(N_min, N_max, N_jump, max_error, beta_min=0.15, beta_max=0.4, gamma_den_min=5, gamma_den_max=14):
    #result = pd.DataFrame(columns=['N', 'beta', 'gamma_den', 'R0', 'error_avg'])
    list = []
    for N in range(N_min, N_max, N_jump):
        situazione_calcolo = (N-N_min) / (N_max-N_min)
        print("Calcolo al: "+str(round(situazione_calcolo*100, 2))+"%")
        for beta in np.arange(beta_min, beta_max, 0.0025):
            for gamma_den in np.arange(gamma_den_min, gamma_den_max, 0.1):
                gamma = 1 / gamma_den
                S, I, R, t = SIR_model(I0, R0, N, beta, gamma)
                positivi_misurati = DataFrame.totale_attualmente_positivi
                guariti_misurati = DataFrame.dimessi_guariti
                error_positivi = (I - positivi_misurati) / positivi_misurati
                error_guariti = (R - guariti_misurati) / guariti_misurati
                error_positivi_avg = statistics.mean(abs(error_positivi))
                error_guariti_avg = statistics.mean(abs(error_guariti))
                if abs(error_positivi_avg) < max_error:
                    #print(N, beta, gamma_den, beta/gamma, error_positivi_avg, error_guariti_avg)
                    list.append([N, beta, gamma_den, beta/gamma, error_positivi_avg, error_guariti_avg])  
    result = pd.DataFrame(list, columns = ["N", "beta", "gamms_den", "R0", "error_positivi_avg", "error_guariti_avg"])
    return result

def plt_SIR_model(t, S, I, R, tempo_misurati, positivi_misurati):
    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w', figsize=(16,10))
    ax = fig.add_subplot(111, axisbelow=True)
    ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.plot(tempo_misurati, positivi_misurati/1000, label='positivi')
    #ax.plot(range(0,len(DataFrame.data)), DataFrame.dimessi_guariti/1000, label='guariti')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number (1000s)')
    #ax.set_ylim(0,1.2)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.savefig("PredictionSIR.jpg")

calibration_flag = 1
#Parameters for calibration
N_min, N_max, N_jump, max_error = 220000, 250000, 2500, 0.2
I0 = 229
R0 = 0
# Calibration
if calibration_flag == 1:
    result = calibration(N_min, N_max, N_jump, max_error, beta_max=0.375)
    index_min_error = result.error_positivi_avg.idxmin()
    N, beta, gamma_den, Rknot, err_positivi, err_guariti = result.iloc[index_min_error]
    print(result.iloc[index_min_error])
print("Calcolo al: 100%")
S, I, R, t  = SIR_model(I0, R0, 160000, 0.345, 1/9, t_max = 120)
S, I, R, t  = SIR_model(I0, R0, N, beta, 1/gamma_den, t_max = 120)
#Plot
positivi_misurati = DataFrame.totale_attualmente_positivi
tempo_misurati = range(0,len(DataFrame.data))
plt_SIR_model(t, S, I, R, tempo_misurati, positivi_misurati)
