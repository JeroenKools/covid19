# Builtins
import datetime
import math
from io import StringIO

# Third party modules
import requests
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import matplotlib.ticker as mtick
from IPython.display import display, Markdown
import numpy as np
import scipy.optimize
from scipy.signal import gaussian

register_matplotlib_converters()
light_grey = (.85, .85, .85, 1)                        # Plot background color
matplotlib.rcParams['figure.figsize'] = (14, 8)        # Default size of all figures
matplotlib.rcParams['axes.facecolor'] = light_grey     # Default background color of all graph areas
matplotlib.rcParams['figure.facecolor'] = light_grey   # Default background color of all figure borders
cm = plt.cm.get_cmap('nipy_spectral')                  # This colormap is used for the colors of the plot lines

# Where to get the data. There have been some issues with the data quality lately. 
# For the most recent data, use branch 'master'.
# For stable March 13 data, use 'c2f5b63f76367505364388f5a189d1012e49e63e'
branch = "master" 
base_url = f"https://raw.githubusercontent.com/CSSEGISandData/COVID-19/{branch}/"+\
            "csse_covid_19_data/csse_covid_19_time_series/"
data_urls = {
    "confirmed": "time_series_19-covid-Confirmed.csv",
    "deaths":    "time_series_19-covid-Deaths.csv",
    "recovered": "time_series_19-covid-Recovered.csv"    
}

from IPython.display import HTML
import random


# Truncate (drop after given number of decimals instead of rounding)
def truncate(n, decimals):
    p = int(10**decimals)
    if decimals > 0:
        return int(math.trunc(p*n)/p)
    else: return int(n)
    

# Format large numbers with k for thousands, M for millions, B for billions
def kmb_number_format(n, digits=4):
    if n <= 0:
        return 0
    decimals = int(digits - (math.log(n) / math.log(10)) % 3) 
    if n < 1e3:
        return f"{truncate(n, decimals)}"
    elif n < 1e6:
        return f"{truncate(n/1e3, decimals)}K"
    elif n < 1e9:
        return f"{truncate(n/1e6, decimals)}M"
    else:
        return f"{truncate(n/1e9, decimals)}B"

    
# Convenience function for labelling the y-axis
def set_y_axis_format(log=True):
    if log:
        plt.yscale("log")
    plt.gca().get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: kmb_number_format(x)))
    
    
def normalize_to_target_product(dist, target):
    ratio = np.product(dist)/target
    ratio = ratio ** (1/len(dist))
    dist = dist / ratio
    return dist


def death_chance_per_day(cfr, sigma, length, do_plot):

    # Approximate survival and death odds
    x = np.arange(length)
    death_chance_per_day = gaussian(length, sigma)
    death_chance_per_day = death_chance_per_day/sum(death_chance_per_day) * cfr
    survive_chance_per_day = 1 - death_chance_per_day

    # Approximation is slightly off, compensate
    survive_chance_per_day = normalize_to_target_product(survive_chance_per_day, 1-cfr)
    death_chance_per_day = 1 - survive_chance_per_day
    alive = np.product(survive_chance_per_day)

    if do_plot:
        display(Markdown(f"Input CFR: {cfr:.2%}. Model result: {alive:.2%} of being alive after {len(x)} days"))
        plt.plot(x,100*death_chance_per_day)
        plt.title("Modelled daily fatality rate, if infected with Covid-19", fontsize=16)
        plt.xlabel("Day")
        plt.ylabel("Chance")
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.grid()
        plt.show()

        plt.plot(x, 100*np.cumprod(survive_chance_per_day))
        plt.title("Modelled survival probability, n days after contracting Covid-19", fontsize=16)
        plt.xlabel("Day")
        plt.ylabel("Chance")
        plt.ylim(0, 100)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.grid()
        plt.show()

    else:
        return death_chance_per_day

    
def logistic_func(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))


def get_pie_label(pct):
    if pct > 1.5:
        return f"{pct/100:1.1%}"
    else:
        return ""