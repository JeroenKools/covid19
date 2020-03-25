# Builtins
import math
import colorsys

# Third party modules
from pandas.plotting import register_matplotlib_converters
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from IPython.display import display, Markdown
import numpy as np
import scipy.stats
from scipy.signal import gaussian

register_matplotlib_converters()
light_grey = (.93, .93, .93, 1)  # Plot background color
matplotlib.rcParams['figure.figsize'] = (14, 8)  # Default size of all figures
matplotlib.rcParams['axes.facecolor'] = light_grey  # Default background color of all graph areas
matplotlib.rcParams['figure.facecolor'] = light_grey  # Default background color of all figure borders
cm = plt.cm.get_cmap('nipy_spectral')  # This colormap is used for the colors of the plot lines

# Where to get the data. There have been some issues with the data quality lately. 
# For the most recent data, use branch 'master'.
# For stable March 13 data, use 'c2f5b63f76367505364388f5a189d1012e49e63e'
branch = "master"
base_url = f"https://raw.githubusercontent.com/CSSEGISandData/COVID-19/{branch}/" + \
           "csse_covid_19_data/csse_covid_19_time_series/"
data_urls = {
    "confirmed": "time_series_covid19_confirmed_global.csv",
    "deaths": "time_series_covid19_deaths_global.csv",
    # No longer being updated: "recovered": "time_series_19-covid-Recovered.csv"
}

continent_codes = {
    "AF": "Africa",
    "AN": "Antarctica",
    "AS": "Asia",
    "EU": "Europe",
    "NA": "North America",
    "OC": "Oceania",
    "SA": "South America"
}


# Truncate (drop after given number of decimals instead of rounding)
def truncate(n, decimals, as_int=True):
    p = int(10 ** decimals)
    if decimals > 0:
        result = math.trunc(p * n) / p
    else:
        result = n
    if as_int or decimals == 0:
        result = int(result)
    else:
        result = round(result, decimals)  # sometimes the result is still off due to float errors
    return result


# Format large numbers with k for thousands, M for millions, B for billions
def kmb_number_format(n, digits=3, as_int=True):
    if n <= 0:
        return 0
    decimals = int(digits - (math.log(n) / math.log(10)) % 3)
    if n < 1e3:
        div = 1
        suffix = ""
        as_int = True
    elif n < 1e6:
        div = 1e3
        suffix = "K"
    elif n < 1e9:
        div = 1e6
        suffix = "M"
    else:
        div = 1e9
        suffix = "B"

    return f"{truncate(n/div, decimals, as_int)}{suffix}"


# Convenience function for labelling the y-axis
def set_y_axis_format(ymax, log=True):
    power = int(math.ceil(math.log(ymax) / math.log(10)))
    if power % 1 > 0.69897:
        ceil = 1.03 * 10 ** power
    else:
        ceil = 1.03 * 10 ** (power - 0.30103)
    plt.ylim(0.98, ceil)

    if log:
        plt.yscale("log", basey=10)
        plt.yticks([float(10 ** x) for x in range(0, power + 1)])

    plt.gca().get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: kmb_number_format(x)))

    plt.minorticks_off()
    plt.gca().tick_params(which="major", color=light_grey)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)


def normalize_to_target_product(dist, target):
    ratio = np.product(dist) / target
    ratio = ratio ** (1 / len(dist))
    dist = dist / ratio
    return dist


def death_chance_per_day(cfr, s=0.9, mu=0, sigma=1, length=20, do_plot=False):
    # Approximate survival and death odds
    x = np.arange(length)
    death_chance = scipy.stats.lognorm.pdf(np.linspace(0, length-1, length), s=s, loc=mu, scale=sigma)

    # Approximation is slightly off, compensate
    if cfr > 0:
        death_chance = death_chance / sum(death_chance) * cfr
        survive_chance_per_day = 1 - death_chance
        survive_chance_per_day = normalize_to_target_product(survive_chance_per_day, 1 - cfr)
        death_chance = 1 - survive_chance_per_day
    else:
        survive_chance_per_day = 1 - death_chance
    alive = np.product(survive_chance_per_day)

    if do_plot:
        display(Markdown(f"Input CFR: {cfr:.2%}. Model result: {alive:.2%} of being alive after {len(x)} days"))
        plt.plot(x, 100 * death_chance)
        plt.title("Modelled daily fatality rate, if infected with Covid-19", fontsize=16)
        plt.xlabel("Day")
        plt.ylabel("Chance")
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.grid()
        plt.show()

        plt.plot(x, 100 * np.cumprod(survive_chance_per_day))
        plt.title("Modelled survival probability, n days after contracting Covid-19", fontsize=16)
        plt.xlabel("Day")
        plt.ylabel("Chance")
        plt.ylim(0, 102)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.grid()
        plt.show()

    else:
        return death_chance


def logistic_func(x, L, k, x0):
    import warnings
    warnings.filterwarnings("error")
    try:
        out = L / (1 + np.exp(-k * (x - x0)))
        return out
    except RuntimeWarning:
        # Ignore these warnings, they will only happen when the curve fit parameters are out of bounds
        return x


def get_pie_label(pct):
    if pct > 1.5:
        return f"{pct / 100:1.1%}"
    else:
        return ""


def string_to_color(name, offset=4):
    fixed_colors = {
        "Netherlands": (1.0, 0.4, 0.0),
        "United States": (0.0, 0.7, 1.0),
        "United Kingdom": (0.6, 0.0, 0.3),
        "Spain": (0.9, 1, 0.0),
        "All except China": (0.2, 0.2, 0.2)
    }

    if name in fixed_colors:
        return fixed_colors[name]

    else:
        hue = 0
        sat = 0.7
        val = 0.7

        for char in " ().-":
            name = name.lower().replace(char, "")
        h = sum([(ord(x) - 97) for x in name]) / 37 % 1
        s = (ord(name[3]) - 97) / 25
        v = (ord(name[2]) - 97) / 25

        c = colorsys.hsv_to_rgb(hue + h,
                                sat + 0.3 * s,
                                val + 0.3 * v)
        return c


def gauss(n=11,sigma=1):
    r = range(-int(n/2), int(n/2)+1)
    return [1 / (sigma * math.sqrt(2*math.pi)) * math.exp(-float(x)**2/(2*sigma**2)) for x in r]


def half_gauss(n=6, sigma=1, ascending=True):
    g = gauss(2*n+1, sigma)
    g = g[:n+1]
    g = g + n * [0]
    s = sum(g)
    if not ascending:
        g = g[::-1]
    return [x/s for x in g]