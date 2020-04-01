# Author: Stefano Bergamini

import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt

from scipy.integrate import odeint

def SIR_model(I0, R0, N, beta, gamma, t_max):
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


def calibration(DataFrame, N, max_error, beta_min=0.15, beta_max=0.45, gamma_den_min=1, gamma_den_max=14):
    # result = pd.DataFrame(columns=['N', 'beta', 'gamma_den', 'R0', 'error_avg'])
    list = []
    t_max = len(DataFrame.data)
    # Initial conditions for Italy COVID19
    I0 = 229
    R0 = 0
    for beta in np.arange(beta_min, beta_max, 0.0025):
        situazione_calcolo = (beta - beta_min) / (beta_max - beta_min)
        print("Calcolo al: " + str(round(situazione_calcolo * 100, 2)) + "%")
        for gamma_den in np.arange(gamma_den_min, gamma_den_max, 0.1):
            gamma = 1 / gamma_den
            S, I, R, t = SIR_model(I0, R0, N, beta, gamma, t_max)
            positivi_misurati = DataFrame.totale_positivi
            guariti_misurati = DataFrame.dimessi_guariti
            error_positivi = (I - positivi_misurati) / positivi_misurati
            error_guariti = (R - guariti_misurati) / guariti_misurati
            error_positivi_avg = statistics.mean(abs(error_positivi))
            error_guariti_avg = statistics.mean(abs(error_guariti))
            if abs(error_positivi_avg) < max_error:
                # print(N, beta, gamma_den, beta/gamma, error_positivi_avg, error_guariti_avg)
                list.append([N, beta, gamma_den, beta / gamma, error_positivi_avg, error_guariti_avg])
    result = pd.DataFrame(list, columns=["N", "beta", "gamms_den", "R0", "error_positivi_avg", "error_guariti_avg"])
    index_min_error = result.error_positivi_avg.idxmin()
    N, beta, gamma_den, Rknot, err_positivi, err_guariti = result.iloc[index_min_error]
    print(result.iloc[index_min_error])
    # Initial conditions for Italy COVID19
    I0 = 229
    R0 = 0
    S, I, R, t = SIR_model(I0, R0, N, beta, 1 / gamma_den, t_max=120)
    positivi_misurati = DataFrame.totale_positivi
    tempo_misurati = range(0, t_max)
    plt_SIR_model(t, S, I, R, tempo_misurati, positivi_misurati, N, beta, gamma_den, Rknot, err_positivi)
    print("Calcolo al: 100%")
    return result


def plt_SIR_model(t, S, I, R, tempo_misurati, positivi_misurati, N, beta, gamma_den, R0, err_positivi):
    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w', figsize=(16, 10))
    ax = fig.add_subplot(111, axisbelow=True)
    ax.plot(t, S / 1000, 'b', alpha=0.5, lw=2, label='Susceptible')
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
    filename = str('PredictionSIR N' + str(round(N)) + ', beta' + str(round(beta, 5)) + ', gamma_den' + str(
        round(gamma_den, 3)) + ', RO' + str(round(R0, 3)) + ', error' + str(round(err_positivi, 4)) + '.jpg')
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
#     positivi_misurati = DataFrame.totale_positivi
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



