# Author: Stefano Bergamini

import pandas as pd
import numpy as np
import multiprocessing as mp
import prediction_sir_model
import prediction_seir_model #first seir model, too slow with too many params
import seir_model
import time

if __name__ == '__main__':

    #Define calibration to do:
    sir_calibration = 0
    seir_calibration = 0
    seir_simple_calibration = 1

    # Import real data
    file_name_it = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
    DataFrame = pd.read_csv(file_name_it)
    t_max = len(DataFrame.data)

    # Plot data
    positivi_misurati = DataFrame.totale_positivi
    tempo_misurati = range(0, len(DataFrame.data))

    if sir_calibration == 1:
        # Parameters for calibration SIR Model
        N_min, N_max, N_jump, max_error = 500000, 1500000, 10000, 0.3

        start = time.time()
        cpu_count = mp.cpu_count()-1
        args = [(DataFrame, N, max_error) for N in range(N_min, N_max, N_jump)]
        with mp.Pool(processes=cpu_count) as p:
            p.starmap(prediction_sir_model.calibration, args)
            # Duration
            finish = time.time()
            timing = finish - start
            print("Execution time: %s  seconds" % (timing))

    if seir_calibration == 1:
        # old model
        # # Parameters for calibration SIR Model
        N_min, N_max, N_jump, max_error = 500000, 1500000, 10000, 0.3
        start = time.time()
        cpu_count = mp.cpu_count()-1
        args = [(DataFrame, N, max_error) for N in range(N_min, N_max, N_jump)]
        with mp.Pool(processes=cpu_count) as p:
            p.starmap(prediction_seir_model.calibration, args)
            # Duration
            finish = time.time()
            timing = finish - start
            print("Execution time: %s  seconds" % (timing))

    if seir_simple_calibration == 1:
        # Parameters for calibration SIR Model
        population, E0, max_error = 60000000, 229, 0.06
        r0_min, r0_max, r0_jump = 3.5, 4.5, 0.05
        daysTotal = 120  # total days to model

        start = time.time()
        cpu_count = mp.cpu_count()-1
        args = [(DataFrame, max_error, r0, population) for r0 in np.arange(r0_min, r0_max, r0_jump)]
        with mp.Pool(processes=cpu_count) as p:
            p.starmap(seir_model.calibration, args)
            # Duration
            finish = time.time()
            timing = finish - start
            print("Execution time: %s  seconds" % (timing))

        #results = calibration(DataFrame, max_error)