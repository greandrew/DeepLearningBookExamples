import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera, kurtosis, skew
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.signal import welch
from scipy.stats import linregress, probplot
from hurst import compute_Hc
import datetime

from scipy.signal import periodogram

def gph(time_series, max_lag = None, min_lag=1):
    if max_lag is None:
        max_lag = len(time_series) // 2
    T = len(time_series)
    n = np.floor(T/2).astype(int)
    lambdaj = 2*np.pi/T*np.arange(1, n+1)
    _, logIj = periodogram(time_series)
    logIj = np.log(logIj[1:])
    Yj = np.log(np.abs(1 - np.exp(1j*lambdaj)))
    Ybar = np.mean(Yj[(min_lag-1):max_lag])
    d_hat = -0.5 * np.sum((Yj[(min_lag-1):max_lag] - Ybar) * logIj[(min_lag-1):max_lag]) / np.sum((Yj[(min_lag-1):max_lag] - Ybar)**2)
    return d_hat

def test_unconditional_distribution(returns):
    jb_stat, jb_pvalue = jarque_bera(returns)
    vol = np.std(returns) * np.sqrt(252)
    kurt = kurtosis(returns)
    skw = skew(returns)
    return jb_stat, jb_pvalue, vol, kurt, skw

def test_volatility_clustering(returns):
    squared_returns = returns ** 2
    absolute_returns = np.abs(returns)
    acf_squared = acf(squared_returns)
    acf_absolute = acf(absolute_returns)
    return acf_squared, acf_absolute

def test_leverage_effect(returns):
    lagged_returns = returns[:-1]
    future_squared_returns = (returns[1:] ** 2)
    cross_correlation = np.correlate(lagged_returns, future_squared_returns)
    return cross_correlation[0]

def test_long_memory(returns):
    squared_returns = returns ** 2
    absolute_returns = np.abs(returns)
    gph_squared = gph(squared_returns)
    gph_absolute = gph(absolute_returns)
    H1, _, _ = compute_Hc(squared_returns)
    H2, _, _ = compute_Hc(absolute_returns)
    return H1, H2, gph_squared, gph_absolute

def run_tests(returns_list, labels):
    results = []

    for returns, label in zip(returns_list, labels):
        jb_stat, jb_pvalue, vol, kurt, skw = test_unconditional_distribution(returns)
        cross_correlation = test_leverage_effect(returns)
        H, _, gph_squared, _ = test_long_memory(returns)

        row_lst = [label, jb_stat, jb_pvalue, vol, kurt, skw,
                        cross_correlation, H, gph_squared]
        results.append(row_lst)

    columns = ['Label', 'JB Stat', 'JB P-value', 'Volatility', 'Kurtosis', 'Skewness',
               'Cross-correlation', 'H Squared', 'GPH Squared']
    df = pd.DataFrame(results, columns=columns)
    df.set_index('Label')
    return df

def plot_acf_volatility_clustering(returns, filepath, name):
    squared_returns = returns ** 2
    absolute_returns = np.abs(returns)

    acf_squared = acf(squared_returns)
    acf_absolute = acf(absolute_returns)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    ax1.bar(range(len(acf_squared)), acf_squared, color='black')
    ax1.set_title("ACF of Squared Returns for " + name)
    ax1.set_xlabel("Lag")
    ax1.set_ylabel("ACF")

    ax2.bar(range(len(acf_absolute)), acf_absolute, color='black')
    ax2.set_title("ACF of Absolute Returns for " + name)
    ax2.set_xlabel("Lag")
    ax2.set_ylabel("ACF")

    plt.tight_layout()

    # Save the plot as a PNG file at 300 DPI
    plt.savefig(filepath, dpi=300)

    plt.show()

def acf_plot(returns, filepath, name):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,6))
    fig.suptitle('ACF Plots of Returns and Squared Returns for ' + name, fontsize=20)

    # Returns ACF Plot
    autocorr_returns = acf(returns)
    ax[0].stem(autocorr_returns, linefmt='k-', markerfmt='ko', basefmt='k-')
    ax[0].set_xlabel('Lag', fontsize=12)
    ax[0].set_ylabel('ACF Returns', fontsize=12)

    # Squared Returns ACF Plot
    autocorr_sq_returns = acf(returns**2)
    ax[1].stem(autocorr_sq_returns, linefmt='k-', markerfmt='ko', basefmt='k-')
    ax[1].set_xlabel('Lag', fontsize=12)
    ax[1].set_ylabel('ACF Squared Returns', fontsize=12)

    # Save the plot as a PNG file at 300 DPI
    plt.savefig(filepath, dpi=300)

    plt.show()

def acf_plots_block(returns_list, labels, filename=None):
    num_plots = len(returns_list)
    fig, ax = plt.subplots(nrows=num_plots, ncols=2, figsize=(10, num_plots*2.5))

    for i in range(num_plots):
        returns = returns_list[i]
        label = labels[i]

        # Returns ACF Plot
        autocorr_returns = acf(returns)
        ax[i,0].stem(autocorr_returns, linefmt='k-', markerfmt='ko', basefmt='k-')
        ax[i,0].set_title(f'ACF Plot of {label} Returns', fontsize=16)
        ax[i,0].set_xlabel('Lag', fontsize=12)
        ax[i,0].set_ylabel('ACF', fontsize=12)

        # Squared Returns ACF Plot
        autocorr_sq_returns = acf(returns**2)
        ax[i,1].stem(autocorr_sq_returns, linefmt='k-', markerfmt='ko', basefmt='k-')
        ax[i,1].set_title(f'ACF Plot of {label} Squared Returns', fontsize=16)
        ax[i,1].set_xlabel('Lag', fontsize=12)
        ax[i,1].set_ylabel('ACF', fontsize=12)

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()

def plot_daily_returns(returns, dates, save_path, labels=None):

    # Convert dates to matplotlib's internal float format
    dates_float = np.array([date.toordinal() for date in dates])
    
    # Set up plot
    fig, ax = plt.subplots(len(returns), 1, figsize=(10, 2*len(returns)))
    if len(returns) == 1:
        ax = [ax]
    fig.subplots_adjust(hspace=0.4)
    
    # Plot each series
    for i, return_series in enumerate(returns):
        ax[i].plot(dates_float, return_series, color='black')
        ax[i].set_ylabel('Returns')
        if labels is not None:
            ax[i].set_title(labels[i])
        ax[i].xaxis.set_major_locator(plt.MaxNLocator(10))
        ax[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: 
            datetime.date.fromordinal(int(x)).strftime('%Y') if pos%365==0 else ''))
    
    # Save plot
    plt.savefig(save_path, dpi=300)
    plt.show()

def qq_plot(real_returns_list, sim_returns_list, save_path, num_cols=2, labels=None):
    num_plots = len(real_returns_list)
    num_rows = int(np.ceil(num_plots / num_cols))
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    quantiles = np.linspace(start=0, stop=1, num=500)
    for i in range(num_plots):
        row = i // num_cols
        col = i % num_cols
        if labels:
            axs[row, col].set_title(labels[i])
        else:
            axs[row, col].set_title(f'QQ plot {i+1}')
        y_quantiles = np.quantile(real_returns_list[i], quantiles, method='nearest')
        x_quantiles = np.quantile(sim_returns_list[i], quantiles, method='nearest')
    
        axs[row, col].plot(x_quantiles, y_quantiles, 'ko')
        axs[row, col].set_xlabel('Simulated Quantiles')
        axs[row, col].set_ylabel('Real Quantiles')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()