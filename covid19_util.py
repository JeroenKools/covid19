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
        return math.trunc(p*n)/p
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

    
def get_dataframes():
    dataframes = {}
    for metric in data_urls.keys():
        url = base_url + data_urls[metric]                             # Combine URL parts
        r = requests.get(url)                                          # Retrieve from URL
        dataframes[metric] = pd.read_csv(StringIO(r.text), sep=",")    # Convert into Pandas dataframe

    # Display the first lines
    display(Markdown("### Raw confirmed cases data, per region/state"))
    with pd.option_context("display.max_rows", 10, "display.max_columns", 14):
        display(dataframes["confirmed"])
    return dataframes


def process(dataframes):
    for metric in data_urls.keys():
        by_country = dataframes[metric].groupby("Country/Region").sum()  # Group by country
        dates = by_country.columns[2:]                                   # Drop Lat/Long columns
        by_country.loc["All except China", dates] =\
            by_country.sum().loc[dates]-by_country.loc["China", dates]   # Add "Outside China" row
        by_country = by_country.loc[:, dates].astype(int)                # Convert to columns to matplotlib dates
        dates = pd.to_datetime(dates) 
        by_country.columns = dates

        if metric == "confirmed":
            # Early China data points
            early_china_data = {
                "1/17/20": 45,
                "1/18/20": 62,
                "1/20/20": 218
            }

            # Insert data points
            for d, n in early_china_data.items():
                by_country.loc["China", pd.to_datetime(d)] = n               

            # Retain chronological column order  
            by_country = by_country.reindex(list(sorted(by_country.columns)), axis=1) 
            by_country = by_country.fillna(0)

            # Correct an odd blip in the Japanese data. 
            # From 2/5 to 2/7, the Johns Hopkins data for Japan goes 22, 45, 25. 
            # I assume that the 45 is incorrect. Replace with 23.5, halfway between the values for 2/5 and 2/7
            by_country.loc["Japan", pd.to_datetime("2/06/20")] = 23.5 

        # Change some weird formal names to more commonly used ones
        by_country = by_country.rename(index={"Republic of Korea": "South Korea", 
                                              "Holy See": "Vatican City",         
                                              "Iran (Islamic Republic of)": "Iran",
                                              "Viet Nam": "Vietnam",
                                              "Taipei and environs": "Taiwan",
                                              "Republic of Moldova": "Moldova",
                                              "Russian Federaration": "Russia",
                                              "Korea, South": "South Korea",
                                              "Taiwan*": "Taiwan"
                                             })        

        # Store processed results for metric
        dataframes[metric+"_by_country"] = by_country

    # Compute active cases
    dataframes["active_by_country"] = dataframes["confirmed_by_country"] - \
                                      dataframes["deaths_by_country"] - \
                                       dataframes["recovered_by_country"]
    
    display(Markdown("### Table of confirmed cases by country"))
    with pd.option_context("display.max_rows", 10, "display.max_columns", 10):
        display(dataframes["confirmed_by_country"])    
    return dataframes


def list_countries(dataframes):
    confirmed_by_country = dataframes["confirmed_by_country"]
    display(Markdown(f"### {len(confirmed_by_country)} countries/territories affected:\n"))
    for i, k in enumerate(confirmed_by_country.index):
        if len(k) > 19:
            k = k[:18] + "."
        print(f"{k:20}", end=" " if (i+1) % 5 else "\n")      # Every 5 items, end with a newline
        
        
def plot(dataframes, x_metric, y_metric, countries_to_plot, colormap=cm, use_log_scale=True, 
         min_cases=40, n_days_average=5):
    
    markers= ["o", "^", "v", "<", ">", "s", "X", "D", "*", "$Y$", "$Z$"]
    short_metric_to_long = {
        "confirmed": "Confirmed cases",
        "deaths":    "Deaths",
        "active":    "Active cases",
        "growth_factor": f"{n_days_average}-day-avg growth factor"
    }
    fills = ["none", "full"] # alternate between filled and empty markers
    length = None
    m = len(markers)
    cm = plt.cm.get_cmap(colormap)
    cNorm = matplotlib.colors.Normalize(vmin=0, vmax=len(countries_to_plot))
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm)
    if y_metric in dataframes:
        by_country   = dataframes[y_metric+"_by_country"]
    elif y_metric == "growth_factor":
        by_country   = dataframes["confirmed_by_country"]
    elif y_metric == "active":
        by_country = dataframes["confirmed_by_country"] - \
                     dataframes["deaths_by_country"] - \
                     dataframes["recovered_by_country"] 
    else:
        print(f"{y_metric}' is an invalid y_metric!")
        
    for i, country in enumerate(countries_to_plot):
        if country not in by_country.index:
            raise KeyError(f"Country '{country}' not found!")
            return
        country_data = by_country.loc[country] # , dates]
        fill = fills[i % (2*m) < m]
        
        if y_metric == "growth_factor":
            if x_metric == "day_number":
                country_data = country_data[country_data >= min_cases]
            country_data = country_data.diff() / country_data + 1
            country_data = np.convolve(country_data, np.ones(n_days_average)/n_days_average, mode="valid")
            
        is_valid = sum(np.nan_to_num(country_data)) > 0
        
        if x_metric == "calendar_date" and is_valid:
            plt.plot(country_data, marker=markers[i%m], label=country, 
                 markersize=6, color=scalarMap.to_rgba(i), alpha=1, fillstyle=fill)
            
        elif x_metric == "day_number":                   
            if y_metric != "growth_factor":
                country_data = country_data[country_data >= min_cases]
            if country == "Outside China":
                length = len(country_data)
            day_nr = list(range(len(country_data)))
            if is_valid:
                plt.plot(day_nr, country_data, marker=markers[i%m], label=country, 
                         markersize=7, color=scalarMap.to_rgba(i), alpha=1, fillstyle=fill)
      
    long_y_metric = short_metric_to_long[y_metric]
    plt.ylabel(long_y_metric, fontsize=14)
    if x_metric == "calendar_date":
        plt.xlabel("Date", fontsize=14)
        plt.title(f"COVID-19 {long_y_metric} over time in selected countries", fontsize=18)
        plt.ylim(0.9*use_log_scale, by_country.loc[countries_to_plot].max().max()*(2-0.9*(not use_log_scale)))
        firstweekday = pd.Timestamp(country_data.index[0]).dayofweek
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1, byweekday=firstweekday))
    elif x_metric == "day_number":
        if y_metric != "growth_factor":        
            floor = 10**math.floor(math.log(min_cases)/math.log(10))
            floor = floor * (1 - (not use_log_scale))  * .9
            ceil  = 10**math.ceil(math.log(by_country.loc[countries_to_plot].max().max())/math.log(10))
            ceil  = ceil * 1.2
            plt.ylim(floor, ceil)            
        plt.xlim(0, length)
        plt.xlabel("Day Number", fontsize=14)
        plt.title(
            f"COVID-19 {long_y_metric}, from the first day with ≥{min_cases} local cases, in selected countries",
            fontsize=18)
        
    plt.legend()
    if y_metric == "growth_factor":
        plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{x:,.2f}"))
        plt.ylabel("Growth Factor", fontsize=14)
    else:
        set_y_axis_format(use_log_scale)
    plt.grid()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().tick_params(which="minor", width=0)
    plt.gca().tick_params(which="major", color=light_grey)    
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.show()
    
    
def get_pie_label(pct):
    if pct > 1.5:
        return f"{pct/100:1.1%}"
    else:
        return ""
    
def plot_pie(dataframes, y_metric):
    short_y = y_metric.split()[0]
    plt.figure(figsize=(8,8))
    data_for_pie = dataframes[short_y+"_by_country"].iloc[:,-1]
    data_for_pie = data_for_pie[data_for_pie.index != "All except China"]
    data_for_pie = data_for_pie.sort_values(ascending=False)    
    countrynames = [x if data_for_pie[x]/data_for_pie.values.sum() > .015 else "" for x in data_for_pie.index]
    data_for_pie.plot.pie(startangle=270, autopct=get_pie_label, labels=countrynames,
                          counterclock=False, pctdistance=.75)

    plt.ylabel("")
    plt.title(f"{y_metric.capitalize()} as of {data_for_pie.name.date()}", fontsize=16)
    plt.show()
    
    
def logistic_func(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))


def curve_fit(dataframes, country="All except China", days=100, do_plot=True):
    x = np.arange(days)
    country_data = dataframes["confirmed_by_country"].loc[country, :]
    country_data = country_data[np.isfinite(country_data)]
    current_day = country_data.index[-1]
    
    [L, k, x0], pcov =  scipy.optimize.curve_fit(logistic_func, np.arange(len(country_data)), 
                                                 country_data, maxfev=10000, 
                                                 p0=[1e6, 0.5, max(1, len(country_data))],
                                                 bounds=([0, 0.0, 1], [1e9, 1.0, 200]),
                                                 method="trf"
                                                )

    # dates up to 100 days after start
    model_date_list = [current_day + datetime.timedelta(days = n) for n in range(0, len(x) - len(country_data))] 
    model_date_list = [mdates.date2num(x) for x in model_date_list]

    n = len(model_date_list)
    sig_L, sig_k, sig_x0 = np.sqrt(np.diag(pcov))
    logistic = logistic_func(x[-n:]-1, L, k, x0)
    logistic_sigma = logistic_func(x[-n:]-1, sig_L, sig_k, sig_x0)
    uncertainty_w = np.linspace(0, 1, len(logistic))

    if do_plot:
        plt.plot(country_data, label="Confirmed cases in " + country, markersize=3, zorder=1)
        plt.plot(model_date_list, 
                 logistic, label=f"{L:.0f} / (1 + e^(-{k:.3f} * (x - {x0:.3f})))", zorder=1)

        plt.grid()
        plt.legend(loc="upper left")
        plt.title(f"Logistic curve fit and extrapolation for {country}", fontsize=18)
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Cases", fontsize=14)
        plt.scatter(mdates.date2num(current_day), country_data[-1], s=20, c="C00", zorder=2)
        plt.annotate(f"{datetime.datetime.strftime(current_day, '%m/%d')}: {country_data[-1]:,.0f}", 
                     (mdates.date2num(current_day)-1, country_data[-1]), fontsize=18, ha="right")

        plt.scatter(model_date_list[-1], logistic[-1], s=20, c="C01", zorder=2)
        plt.annotate(f"{mdates.num2date(model_date_list[-1]).strftime('%m/%d')}: {logistic[-1]:,.0f}", 
                     (model_date_list[-1]-1, logistic[-1]*1.08), fontsize=18, ha="right")
        set_y_axis_format(True)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.gca().tick_params(which="both", color=light_grey)
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.show()
        
      
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
    
    
def simulate_country_history(dataframes, country, population, history_length=28, show_result=False):    
    confirmed = dataframes["confirmed_by_country"].loc[country]
    deaths = dataframes["deaths_by_country"].loc[country]
    recovered = dataframes["recovered_by_country"].loc[country]
    active = confirmed - deaths - recovered
    uninfected = population - confirmed[-1] - deaths[-1] - recovered[-1 ] 
    
    simulation = pd.DataFrame(data=[confirmed, deaths, recovered, active], 
                              index=["confirmed", "deaths", "recovered", "active"]).transpose()
    simulation = simulation.fillna(0)
    
    # reconstruct active case durations by assuming FIFO
    for i, day in enumerate(confirmed.index):
        case_history = [0] * history_length
        if i == 0:
            case_history[0] = simulation.confirmed[0]
        else:
            new_cases     = simulation.confirmed.diff()[i]
            new_deaths    = simulation.deaths.diff()[i]
            new_recovered = simulation.recovered.diff()[i]
            newly_resolved = new_deaths + new_recovered
                        
            case_history[0] = new_cases
            case_history[1:] = simulation.iloc[i-1, -history_length:]
            case_history = case_history[:history_length]

        for h in range(history_length):
            x = history_length - h - 1
            #print(h, x, case_history)
            oldest_active = case_history[x]
            if i != 0 and newly_resolved > 0:  
                if oldest_active >= newly_resolved: 
                    case_history[x] = oldest_active - newly_resolved
                    newly_resolved = 0
                else:
                    newly_resolved -= oldest_active
                    case_history[x] = 0                    

            simulation.at[day.to_datetime64(), f"active_{h}"] = case_history[h]
        
    simulation = simulation.fillna(0).astype(int)
    if show_result:
        display(Markdown("<br>**First 10 days in the US, showing a 7-day case duration history:**"))
        display(simulation.iloc[:10, :])
    return simulation


def simulate_country(
     dataframes,
     country,                      # name of the country to simulate
     population,                   # population of the country
     days=30,                      # how many days into the future to simulate
     cfr=0.02,                     # case fatality rate, 0 to 1
     history_length=28,            # length of case history
     mean_death_day=14,            # How many days after infection people are most likely to die
     sigma_death_days=5,           # Standard deviation in mortality over time distribution
     growth_rate_trend=[1.2, 0.8]  # Growth factor development over time. This will be linearly
                                   # interpolated to a vector of length {days}
    ):
    growth_rate_per_day = np.interp(np.linspace(0,1,days), 
                                    np.linspace(0,1,len(growth_rate_trend)), 
                                    growth_rate_trend)
                     
    country_history = simulate_country_history(dataframes, country, population, history_length)
    daily_death_chance = death_chance_per_day(cfr, sigma_death_days, history_length, do_plot=False)
    today = country_history.index[-1]
    
    for d in range(days):        
        # column shortcuts
        confirmed = country_history.confirmed
        deaths = country_history.deaths
        active = country_history.active
        recovered = country_history.recovered
        uninfected = population - confirmed.iloc[-1] - deaths.iloc[-1] - active.iloc[-1] - recovered.iloc[-1]
        case_history = country_history.iloc[-1, -history_length:].copy()
        
        last_day = confirmed.index[-1]
        next_day = last_day + pd.DateOffset(1)
        daily_growth = growth_rate_per_day[d]
        last_delta = confirmed[-1] - confirmed[-2]
        
        
        # Infect
        # TODO: Use R0 and base new cases on active cases, rather than 
        #       growth factor and new cases based on new cases the day before
        last_confirmed = confirmed.loc[last_day]
        uninfected_ratio = uninfected / (uninfected + active[-1])
        new_cases = int(np.maximum(0, last_delta * daily_growth * uninfected_ratio))
        
        # Deaths
        new_deaths = 0
        for case_duration in range(history_length):
            p = daily_death_chance[case_duration]
            deaths_for_duration = np.random.binomial(case_history[case_duration], 
                                                     daily_death_chance[case_duration])
            case_history[case_duration] -= deaths_for_duration
            new_deaths += deaths_for_duration
        
        # Recoveries
        new_recovered = case_history[-1]
        
        # Uninfected
        uninfected = int(np.maximum(0, uninfected - new_cases - new_deaths - new_recovered))
        
        # Shift case history
        case_history[1:] = case_history[:-1]
        case_history.iloc[0] = new_cases
        
        country_history.at[next_day, "confirmed"] = last_confirmed + new_cases
        country_history.at[next_day, "deaths"] = deaths.loc[last_day] + new_deaths
        country_history.at[next_day, "recovered"] = recovered.loc[last_day] + new_recovered
        country_history.at[next_day, "active"] = active.loc[last_day] + new_cases - new_deaths - new_recovered
        country_history.iloc[-1, -history_length:] = case_history
        
    return country_history, today

def plot_simulation(dataframes, country, days):
    simulation, today = simulate_country(dataframes, country=country, population=330e6, days=days, 
                                     growth_rate_trend=[1.25, 1.05, 0.85, 0.75])

    for metric in ["confirmed cases", "deaths", "active cases", "recovered cases"]:
        short_metric = metric.split()[0]
        plt.plot(simulation.loc[:today, short_metric], c="C00", label="Actual")
        plt.plot(simulation.loc[today:, short_metric], c="C01", label="Simulated")
        plt.title(f"Simulation of {metric} of COVID-19 in {country} for the next {days} days", fontsize=16)
        set_y_axis_format(False)
        plt.grid()
        plt.legend(loc="upper left")
        plt.show()