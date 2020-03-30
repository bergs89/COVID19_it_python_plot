# Author: Stefano Bergamini

import pandas as pd
import numpy as np
import multiprocessing as mp
import prediction_sir_model
import prediction_seir_model

if __name__ == '__main__':
    # Import real data
    file_name_it = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
    file_name_regions = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv"
    DataFrame = pd.read_csv(file_name_it)
    DataFrame_regions = pd.read_csv(file_name_regions)
    t_max = len(DataFrame.data)
    # Plot data
    positivi_misurati = DataFrame.totale_attualmente_positivi
    tempo_misurati = range(0, len(DataFrame.data))
    # Parameters for calibration SIR Model
    N_min, N_max, N_jump, max_error = 360000, 2000000, 5000, 0.3
    processes = [mp.Process(target=prediction_sir_model.calibration, args=(DataFrame, N, max_error,)) for N in range(N_min, N_max, N_jump)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    # SEIR model
    N_min, N_max, N_jump, max_error = 360000, 750000, 10000, 0.3
    processes = [mp.Process(target=prediction_seir_model.calibration, args=(DataFrame, N, max_error,)) for N in
                 range(N_min, N_max, N_jump)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()