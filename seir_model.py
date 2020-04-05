import math
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import matplotlib.widgets  # Cursor
import matplotlib.dates
import statistics
import pandas as pd

# Initialization
population = 60000000

E0 = 229  # exposed at initial time step
daysTotal = 120  # total days to model

days0 = 14  # Germany:57 France: Italy:62? Spain:68? 'all'butChina:65? days before lockdown measures - you might need to adjust this according to output "lockdown measures start:"
days1 = 25
days2 = 34
days3 = 41
# r0 = 2.9  # https://en.wikipedia.org/wiki/Basic_reproduction_number
r0 = 4
r1 = 2.3
r2 = 1.6
r3 = 1.285
rn = 1.0  # reproduction number after quarantine measures - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3539694
          # it seems likely that measures will become more restrictive if r1 is not small enough

timePresymptomatic = 2.5  # almost half infections take place before symptom onset (Drosten) https://www.medrxiv.org/content/10.1101/2020.03.08.20032946v1.full.pdf
# I in this model is maybe better described as 'Infectors'? Event infectious persons in quarantine do not count.
sigma = 1.0 / (5.2 - timePresymptomatic)  # The rate at which an exposed person becomes infectious.  symptom onset - presympomatic
# for SEIR: generationTime = 1/sigma + 0.5 * 1/gamma = timeFromInfectionToInfectiousness + timeInfectious  https://en.wikipedia.org/wiki/Serial_interval

generationTime = 4.6  # https://www.medrxiv.org/content/10.1101/2020.03.05.20031815v1  http://www.cidrap.umn.edu/news-perspective/2020/03/short-time-between-serial-covid-19-cases-may-hinder-containment
gamma = 1.0 / (2.0 * (generationTime - 1.0 / sigma))  # The rate an infectious is not recovers and moves into the resistant phase. Note that for the model it only means he does not infect anybody any more.

beta0 = r0 * gamma  # The parameter controlling how often a susceptible-infected contact results in a new infection.
beta1 = r1 * gamma  # beta0 is used during days0 phase, beta1 after days0
beta2 = r2 * gamma
beta3 = r3 * gamma
betan = rn * gamma

def SEIR_simple_model(Y, x, N, days0, days1, days2, days3, beta0, beta1, beta2, beta3, betan, gamma, sigma):
    # :param array x: Time step (days)
    # :param int N: Population
    # :param float beta: The parameter controlling how often a susceptible-infected contact results in a new infection.
    # :param float gamma: The rate an infected recovers and moves into the resistant phase.
    # :param float sigma: The rate at which an exposed person becomes infective.

    S, E, I, R = Y

    # beta = beta0 if x < days0 else if (x > days0 & x < days1) beta1 else if x > days1 beta2
    if x < days0:
        beta = beta0
    elif (x >= int(days0)) & (x < int(days1)):
        beta = beta1
    elif (x >= int(days1)) & (x < int(days2)):
        beta = beta2
    elif (x >= int(days2)) & (x < int(days3)):
        beta = beta3
    else:
        beta = betan

    dS = - beta * S * I / N
    dE = beta * S * I / N - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I
    return dS, dE, dI, dR

def solveSEIRsimple(SEIR_simple_model, population, E0, days0, days1, days2, days3, beta0, beta1, beta2, beta3, betan, gamma, sigma, daysTotal=120):
    X = np.arange(daysTotal)  # time steps array
    N0 = population - E0, E0, 0, 0  # S, E, I, R at initial step

    y_data_var = scipy.integrate.odeint(SEIR_simple_model, N0, X, args=(population, days0, days1, days2, days3, beta0, beta1, beta2, beta3, betan, gamma, sigma))

    S, E, I, R = y_data_var.T  # transpose and unpack
    return X, S, E, I, R  # note these are all arrays

# Example:
# X, S, E, I, R = solveSEIRsimple(SEIR_simple_model, population, E0, days0, days1, days2, days3, beta0, beta1, beta2, beta3, betan, gamma, sigma)

def plt_SEIR_model(X, S, I, E, R, tempo_misurati, positivi_misurati):
    fig = plt.figure(dpi=75, figsize=(28,8))
    ax = fig.add_subplot(111)
    ax.fmt_xdata = matplotlib.dates.DateFormatter('%Y-%m-%d')  # higher date precision for cursor display
    #ax.plot(X, S, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(X, E, 'y', alpha=0.5, lw=2, label='Exposed (realtime)')
    ax.plot(X, I, 'r--', alpha=0.5, lw=1, label='Infected (realtime)')
    ax.plot(tempo_misurati, positivi_misurati, label='positivi')
    ax.legend()
    ax.grid()
    plt.show()

def calibration(DataFrame, max_error, population=60000000, E0 = 229, days0=14, days1=25, days2=34, days3=41, rmin=1, rmax=4, rjump=0.15):
    # result = pd.DataFrame(columns=['N', 'beta', 'gamma_den', 'R0', 'error_avg'])
    list = []
    t_max = len(DataFrame.data)
    positivi_misurati = DataFrame.totale_positivi
    tempo_misurati = range(0, t_max)
    # Initial conditions for Italy COVID19
    for r0 in np.arange(rmin, rmax, rjump):
        situazione_calcolo = (r0 - rmin) / (rmax - rmin)
        print("Calcolo al: " + str(round(situazione_calcolo * 100, 2)) + "%")
        for r1 in np.arange(rmin, rmax, rjump):
            for r2 in np.arange(rmin, rmax, rjump):
                for r3 in np.arange(rmin, rmax, rjump):
                    # r0 = 4
                    # r1 = 2.3
                    # r2 = 1.6
                    # r3 = 1.285
                    rn = 1.0

                    timePresymptomatic = 2.5
                    sigma = 1.0 / (5.2 - timePresymptomatic)
                    generationTime = 4.6
                    gamma = 1.0 / (2.0 * (generationTime - 1.0 / sigma))
                    
                    beta0 = r0 * gamma
                    beta1 = r1 * gamma
                    beta2 = r2 * gamma
                    beta3 = r3 * gamma
                    betan = rn * gamma
                    
                    X, S, E, I, R = solveSEIRsimple(SEIR_simple_model, population, E0, days0, days1, days2, days3, beta0, beta1,
                                                    beta2, beta3, betan, gamma, sigma)
                    error_positivi_avg = np.average((pow(pow((pd.Series(I) - positivi_misurati),2),1/2)/positivi_misurati).fillna(0))
                    if abs(error_positivi_avg) < max_error:
                        # print(N, beta, gamma_den, beta/gamma, error_positivi_avg, error_guariti_avg)
                        list.append([population, beta0, beta1, beta2, beta3, betan, gamma, sigma, error_positivi_avg])
    result = pd.DataFrame(list, columns=["N", "beta0", "beta1", "beta2", "beta3", "betan", "gamma", "sigma", "error_positivi_avg"])
    index_min_error = result.error_positivi_avg.idxmin()
    N, beta0, beta1, beta2, beta3, betan, gamma, sigma, err_positivi = result.iloc[index_min_error]
    print(result.iloc[index_min_error])
    # Initial conditions for Italy COVID19
    I0 = 229
    X, S, E, I, R = solveSEIRsimple(SEIR_simple_model, population, E0, days0, days1, days2, days3, beta0, beta1, beta2, beta3, betan, gamma, sigma)
    plt_SEIR_model(X, S, I, E, R, tempo_misurati, positivi_misurati)
    print("Calcolo al: 100%")
    return result

file_name_it = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
DataFrame = pd.read_csv(file_name_it)
t_max = len(DataFrame.data)
max_error = 0.12
results = calibration(DataFrame, max_error)