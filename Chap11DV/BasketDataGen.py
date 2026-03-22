import numpy as np
from scipy.stats import beta
import QuantLib as ql
import numpy as np
import pandas as pd
import random as rd
import os
import multiprocessing as mp
import time

def generate_vine_correlation(d, b1, b2):
    P = np.zeros((d, d))  # storing partial correlations
    S = ql.Matrix(d,d,0)
    for j in range(d):
        S[j][j] = 1.0
    for k in range(d-1):
        for i in range(k+1, d):
            P[k, i] = beta.rvs(b1, b2)  # sampling from beta
            P[k, i] = (P[k, i] - 0.5) * 2  # linearly shifting to [-1, 1]
            p = P[k, i]
            for l in range(k-1, -1, -1):  # converting partial correlation to raw correlation
                p = p * np.sqrt((1 - P[l, i]**2) * (1 - P[l, k]**2)) + P[l, i] * P[l, k]
            S[k][i] = p
            S[i][k] = p
    return S

def value_basket(interest_rate, maturity, strike, stocks, vols, divs, corr_matrix, mc_samples_size):
    todays_date = ql.Date(4, 11, 2023)
    ql.Settings.instance().evaluationDate = todays_date
    basket_size = len(stocks)

    settlement_date = todays_date 
    risk_free_rate = ql.FlatForward(settlement_date, interest_rate, ql.Actual365Fixed())

    # option parameters
    exercise = ql.EuropeanExercise(todays_date + maturity)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)

    # market data
    underlying = [ql.SimpleQuote(stock) for stock in stocks]
    volatility = [ql.BlackConstantVol(todays_date, ql.TARGET(), vol, ql.Actual365Fixed()) for vol in vols]
    dividend_yield = [ql.FlatForward(settlement_date, div, ql.Actual365Fixed()) for div in divs]

    processes = [ql.BlackScholesMertonProcess(ql.QuoteHandle(underlying[i]),\
                              ql.YieldTermStructureHandle(dividend_yield[i]),\
                              ql.YieldTermStructureHandle(risk_free_rate),\
                              ql.BlackVolTermStructureHandle(volatility[i])) for i in range(basket_size)]
    
    process = ql.StochasticProcessArray(processes, corr_matrix)
    basketoption = ql.BasketOption(ql.MinBasketPayoff(payoff), exercise)

    result_lst = list()
    for sample_size in mc_samples_size:

        basket_engine = ql.MCEuropeanBasketEngine(process,'PseudoRandom', timeSteps = 1,\
                                           requiredSamples = sample_size, maxSamples = sample_size)
        basketoption.setPricingEngine(basket_engine)

        start_time = time.process_time()
        try:
            option_value = basketoption.NPV()
            error_estimate = basketoption.errorEstimate()

            end_time = time.process_time()
            elapsed_time = end_time - start_time

            mydata= {'maturity':[float(maturity)]}

            for i in range(basket_size):
                stock_name = 'stock_price' + str(i)
                vol_name = 'vols' + str(i)
                mydata[stock_name] = stocks[i]
                mydata[vol_name] = vols[i]

            for j in range(basket_size-1):
                for k in range(j+1,basket_size):
                    rhoName = 'rho_' + str(j) + '_' + str(k)
                    mydata[rhoName] = corr_matrix[j][k]

                mydata['option_value'] = [float(option_value)]
                mydata['error_estimate'] = [float(error_estimate)]
                mydata['samples'] = [int(sample_size)]
                mydata['process_time'] = [float(elapsed_time)]                
            result_lst.append(mydata)

        except Exception as myException:
            end_time = time.process_time()
            elapsed_time = end_time - start_time
            print("Error: {0}".format(myException))

    return result_lst

def generate_basket_data_impl(
    worker_id,
    samples,
    filename,
    mc_samples_lst,
    stock_mean,
    stock_vol,
    b1,
    b2,
    basket_size,
    vol_max=1.0,
    stock_mode="lognormal",
    stock_max=400.0,
):
    rows = []
    strike = 100

    for _ in range(samples):
        maturity = int(rd.uniform(1, 43) ** 2.0)
        interest_rate = 0.0

        if stock_mode == "lognormal":
            stocks = [
                100.0 * rd.lognormvariate(stock_mean, stock_vol)
                for _ in range(basket_size)
            ]
        else:
            stocks = [rd.uniform(0.0, stock_max) for _ in range(basket_size)]

        vols = [rd.uniform(0.0, vol_max) for _ in range(basket_size)]
        divs = [0.0 for _ in range(basket_size)]
        corr_matrix = generate_vine_correlation(basket_size, b1, b2)

        result_lst = value_basket(
            interest_rate,
            maturity,
            strike,
            stocks,
            vols,
            divs,
            corr_matrix,
            mc_samples_lst,
        )

        rows.extend(result_lst)

    df = pd.DataFrame(rows)
    csv_file = f"{filename}{worker_id}.csv"
    df.to_csv(csv_file, index=False)
    return worker_id

def generate_basket_data(threads, samples, filename, mc_samples_lst, stock_mean, stock_vol, b1, b2, basket_size, mat_max = 43, vol_max = 1.0):
    rd.seed()
    pool = mp.Pool(processes=threads)
    block_size = samples // threads
    results = [pool.apply_async(generate_basket_data_impl, 
                                args=(i,block_size, filename, mc_samples_lst, 
                                stock_mean, stock_vol, b1, b2, basket_size, mat_max, vol_max)) for i in range(threads)]
    output = [p.get() for p in results]
    print(output)