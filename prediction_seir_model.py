# Author: Stefano Bergamini
# Source: https://github.com/coronafighter/coronaSEIR/blob/master/main_coronaSEIR.py

import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt

import math
import matplotlib.widgets  # Cursor
import matplotlib.dates

from scipy.integrate import odeint

# Introduction to simplify
days0 = 60  # Germany:60 France: Italy:65? Spain:71? 'all'butChina:68? days before lockdown measures - you might need to adjust this according to output "lockdown measures start:"

r0 = 3.0  # https://en.wikipedia.org/wiki/Basic_reproduction_number
r1 = 1.0  # reproduction number after quarantine measures - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3539694
          # it seems likely that measures will become more restrictive if r1 is not small enough

timePresymptomatic = 2.5  # almost half infections take place before symptom onset (Drosten) https://www.medrxiv.org/content/10.1101/2020.03.08.20032946v1.full.pdf

# I in this model is maybe better described as 'Infectors'? Event infectious persons in quarantine do not count.
sigma = 1.0 / (5.2 - timePresymptomatic)  # The rate at which an exposed person becomes infectious.  symptom onset - presympomatic
# for SEIR: generationTime = 1/sigma + 0.5 * 1/gamma = timeFromInfectionToInfectiousness + timeInfectious  https://en.wikipedia.org/wiki/Serial_interval
generationTime = 4.6  # https://www.medrxiv.org/content/10.1101/2020.03.05.20031815v1  http://www.cidrap.umn.edu/news-perspective/2020/03/short-time-between-serial-covid-19-cases-may-hinder-containment
gamma = 1.0 / (2.0 * (generationTime - 1.0 / sigma))  # The rate an infectious is not recovers and moves into the resistant phase. Note that for the model it only means he does not infect anybody any more.

noSymptoms = 0.35  # https://www.zmescience.com/medicine/iceland-testing-covid-19-0523/  but virus can already be found in throat 2.5 days before symptoms (Drosten)
findRatio = (1.0 - noSymptoms) / 2.0  # wild guess! italy:8? germany:2 south korea < 1???  a lot of the mild cases will go undetected  assuming 100% correct tests

timeInHospital = 12
timeInfected = 1.0 / gamma  # better timeInfectious?

# lag, whole days - need sources
presymptomaticLag = round(timePresymptomatic)  # effort probably not worth to be more precise than 1 day
communicationLag = 2
testLag = 3
symptomToHospitalLag = 5
hospitalToIcuLag = 5

infectionFatalityRateA = 0.005  # Diamond Princess, age corrected plus some optimism
infectionFatalityRateB = infectionFatalityRateA * 3.0  # higher lethality without ICU - by how much?  even higher without oxygen and meds
icuRate = infectionFatalityRateA * 2  # Imperial College NPI study: hospitalized/ICU/fatal = 6/2/1

beta0 = r0 * gamma  # The parameter controlling how often a susceptible-infected contact results in a new infection.
beta1 = r1 * gamma  # beta0 is used during days0 phase, beta1 after days0

s1 = 0.5 * (-(sigma + gamma) + math.sqrt((sigma + gamma) ** 2 + 4 * sigma * gamma * (r0 -1)))  # https://hal.archives-ouvertes.fr/hal-00657584/document page 13
doublingTime = (math.log(2.0, math.e) / s1)

# Model
def SEIR_model(E0, I0, R0, N, beta, gamma, lambd, mu, alpha, t_max):
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - E0 - I0 - R0
    # A grid of time points (in days)
    t = np.linspace(0, t_max, t_max)

    # The SIR model differential equations.
    def deriv(y, t, N, beta, gamma, lambd, mu, alpha):
        S, E, I, R = y
        dSdt = (lambd - mu) * S - beta * S * I / N
        dEdt = beta * S * I / N - (mu + alpha) * E
        dIdt = beta * S * I / N - (gamma + mu) * I
        dRdt = gamma * I - mu * R
        return dSdt, dEdt, dIdt, dRdt

    # Initial conditions vector
    y0 = S0, E0, I0, R0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma, lambd, mu, alpha))
    S, E, I, R = ret.T
    return S, E, I, R, t


N = 500000
#beta = 0.45
#gamma = 1/8
lambd = mu = alpha = 1
t_max = 120

def calibration(DataFrame, N, max_error, beta_min=0.15, beta_max=0.6, gamma_den_min=6, gamma_den_max=12):
    # result = pd.DataFrame(columns=['N', 'beta', 'gamma_den', 'R0', 'error_avg'])
    list = []
    t_max = len(DataFrame.data)
    # Initial conditions for Italy COVID19
    E0 = I0 = 229
    R0 = 0
    # Strumenti per calcolo errore
    positivi_misurati = DataFrame.totale_attualmente_positivi
    guariti_misurati = DataFrame.dimessi_guariti
    # Cicli di calibrazione
    for beta in np.arange(beta_min, beta_max, 0.05):
        situazione_calcolo = (beta - beta_min) / (beta_max - beta_min)
        print("Calcolo al: " + str(round(situazione_calcolo * 100, 2)) + "%")
        for gamma_den in np.arange(gamma_den_min, gamma_den_max, 0.5):
            gamma = 1 / gamma_den
            for lambd in np.arange(0.25, 2, 0.15):
                for mu in np.arange(0.25, 2, 0.15):
                    for alpha in np.arange(0.25, 2, 0.15):
                        S, E, I, R, t = SEIR_model(E0, I0, R0, N, beta, gamma, lambd, mu, alpha, t_max)
                        error_positivi = (I - positivi_misurati) / positivi_misurati
                        error_guariti = (R - guariti_misurati) / guariti_misurati
                        error_positivi_avg = statistics.mean(abs(error_positivi))
                        error_guariti_avg = statistics.mean(abs(error_guariti))
                        if abs(error_positivi_avg) < max_error:
                            # print(N, beta, gamma_den, beta/gamma, error_positivi_avg, error_guariti_avg)
                            list.append([N, beta, gamma_den, beta / gamma, lambd, mu, alpha, error_positivi_avg, error_guariti_avg])
    result = pd.DataFrame(list, columns=["N", "beta", "gamms_den", "R0", "lambd", "mu", "alpha", "error_positivi_avg", "error_guariti_avg"])
    index_min_error = result.error_positivi_avg.idxmin()
    N, beta, gamma_den, Rknot, lambd, mu, alpha, err_positivi, err_guariti = result.iloc[index_min_error]
    print(result.iloc[index_min_error])
    gamma = 1/gamma_den
    # Initial conditions for Italy COVID19
    I0 = 229
    R0 = 0
    S, E, I, R, t = SEIR_model(E0, I0, R0, N, beta, gamma, lambd, mu, alpha, t_max)
    tempo_misurati = range(0, t_max)
    plt_SEIR_model(t, S, E, I, R, tempo_misurati, positivi_misurati, N, beta, gamma_den, Rknot, lambd, mu, alpha, err_positivi)
    print("Calcolo al: 100%")
    return result


def plt_SEIR_model(t, S, E, I, R, tempo_misurati, positivi_misurati, N, beta, gamma_den, R0, lambd, mu, alpha, err_positivi):
    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w', figsize=(16, 10))
    ax = fig.add_subplot(111, axisbelow=True)
    ax.plot(t, S / 1000, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, E / 1000, 'b', alpha=0.5, lw=2, label='Exposed')
    ax.plot(t, I / 1000, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R / 1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.plot(tempo_misurati, positivi_misurati / 1000, label='positivi')
    # ax.plot(range(0,len(DataFrame.data)), DataFrame.dimessi_guariti/1000, label='guariti')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number (1000s)')
    # ax.set_ylim(0,1.2)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    filename = str('PredictionSEIR N' + str(round(N)) + ', beta' + str(round(beta, 5)) + ', gamma_den' + str(
        round(gamma_den, 3)) + ', RO' + str(round(R0, 3)) + ', lambda' + str(round(lambd, 3)) + ', mu' + str(round(mu, 3)) + ', alpha' + str(round(alpha, 3)) + ', error' + str(round(err_positivi, 4)) + '.jpg')
    plt.savefig(filename)
    return

# Example
# import prediction_sir_model
#
# if __name__ == '__main__':
#     # Import real data
#     file_name_it = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
#     file_name_regions = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv"
#     DataFrame = pd.read_csv(file_name_it)
#     DataFrame_regions = pd.read_csv(file_name_regions)
#
#     # Plot data
#     positivi_misurati = DataFrame.totale_attualmente_positivi
#     tempo_misurati = range(0, len(DataFrame.data))
#
#     # Parameters for calibration
#     N_min, N_max, N_jump, max_error = 360000, 600000, 5000, 0.3
#     I0 = 229
#     R0 = 0
#     processes = [mp.Process(target=calibration, args=(DataFrame, N, max_error,)) for N in range(N_min, N_max, N_jump)]
#     for p in processes:
#       p.start()
#     for p in processes:
#       p.join()
