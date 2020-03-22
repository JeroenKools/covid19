from covid19_util import *

from matplotlib import dates as mdates
import pandas as pd
import requests
import scipy.optimize
from io import StringIO
import datetime


class Covid19Processing:
    def __init__(self):
        self.dataframes = {}
        for metric in data_urls.keys():
            url = base_url + data_urls[metric]  # Combine URL parts
            r = requests.get(url)  # Retrieve from URL
            self.dataframes[metric] = pd.read_csv(StringIO(r.text), sep=",")  # Convert into Pandas dataframe

        # Display the first lines
        display(Markdown("### Raw confirmed cases data, per region/state"))
        with pd.option_context("display.max_rows", 10, "display.max_columns", 14):
            display(self.dataframes["confirmed"])

    def process(self):
        for metric in data_urls.keys():
            by_country = self.dataframes[metric].groupby("Country/Region").sum()  # Group by country
            dates = by_country.columns[2:]  # Drop Lat/Long columns
            by_country.loc["All except China", dates] = \
                by_country.sum().loc[dates] - by_country.loc["China", dates]  # Add "Outside China" row
            by_country = by_country.loc[:, dates].astype(int)  # Convert to columns to matplotlib dates
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

            if metric == "recovered":
                # US recovered per state is 0 for these days, while nationwide data is not in the time series csv at all
                # See here: https://github.com/CSSEGISandData/COVID-19/issues/1113
                by_country.loc["US", pd.to_datetime("3/18/20")] = 106
                by_country.loc["US", pd.to_datetime("3/19/20")] = 108
                by_country.loc["US", pd.to_datetime("3/20/20")] = 147
                by_country.loc["US", pd.to_datetime("3/21/20")] = 171

                # Change some weird formal names to more commonly used ones
            by_country = by_country.rename(index={"Republic of Korea": "South Korea",
                                                  "Holy See": "Vatican City",
                                                  "Iran (Islamic Republic of)": "Iran",
                                                  "Viet Nam": "Vietnam",
                                                  "Taipei and environs": "Taiwan",
                                                  "Republic of Moldova": "Moldova",
                                                  "Russian Federaration": "Russia",
                                                  "Korea, South": "South Korea",
                                                  "Taiwan*": "Taiwan",
                                                  "occupied Palestinian territory": "Palestine"
                                                  })

            # Store processed results for metric
            self.dataframes[metric + "_by_country"] = by_country

        # Compute active cases
        self.dataframes["active_by_country"] = self.dataframes["confirmed_by_country"] - \
                                               self.dataframes["deaths_by_country"] - \
                                               self.dataframes["recovered_by_country"]

        display(Markdown("### Table of confirmed cases by country"))
        with pd.option_context("display.max_rows", 10, "display.max_columns", 10):
            display(self.dataframes["confirmed_by_country"])

    def list_countries(self, columns=5):
        confirmed_by_country = self.dataframes["confirmed_by_country"]
        n_countries = len(confirmed_by_country)
        display(Markdown(f"### {n_countries} countries/territories affected:\n"))
        for i, k in enumerate(confirmed_by_country.index):
            if len(k) > 19:
                k = k[:18].strip() + "."
            print(f"{k:20}", end=" " if (i + 1) % columns else "\n")  # Every 5 items, end with a newline

    def get_country_data(self, metric):
        return self.dataframes[metric + "_by_country"]

    def get_new_cases_details(self, country, avg_n=5, median_n=3):
        confirmed = self.dataframes["confirmed_by_country"].loc[country]
        deaths = self.dataframes["deaths_by_country"].loc[country]
        df = pd.DataFrame(confirmed)
        df = df.rename(columns={country: "confirmed_cases"})
        df.loc[:, "new_cases"] = np.maximum(0, confirmed.diff())
        df.loc[:, "new_deaths"] = np.maximum(0, deaths.diff())
        df = df.loc[df.new_cases > 1, :]
        df.loc[:, "growth_factor"] = df.new_cases.diff() / df.new_cases.shift(1) + 1
        df[~np.isfinite(df)] = np.nan
        df.loc[:, "filtered_new_cases"] = \
            scipy.ndimage.convolve(df.new_cases, np.ones(avg_n) / avg_n, origin=-avg_n // 2 + 1)
        df.loc[:, "filtered_growth_factor"] = \
            df.filtered_new_cases.diff() / df.filtered_new_cases.shift(1) + 1
        df.filtered_growth_factor = scipy.ndimage.median_filter(df.filtered_growth_factor, median_n, mode="nearest")
        return df

    def plot(self, x_metric, y_metric, countries_to_plot, colormap=cm, use_log_scale=True,
             min_cases=0, sigma=5, fixed_country_colors=True):

        # layout/style stuff
        markers = ["o", "^", "v", "<", ">", "s", "X", "D", "*", "$Y$", "$Z$"]
        short_metric_to_long = {
            "confirmed": "Confirmed cases",
            "deaths": "Deaths",
            "active": "Active cases",
            "growth_factor": f"{sigma}-day avg growth factor",
            "deaths/confirmed": "Case Fatality Rate"
        }
        fills = ["none", "full"]  # alternate between filled and empty markers
        length = None
        m = len(markers)
        cm = plt.cm.get_cmap(colormap)
        cNorm = matplotlib.colors.Normalize(vmin=0, vmax=len(countries_to_plot))
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm)

        ratio_parts = y_metric.split("/")
        if y_metric in self.dataframes:
            by_country = self.get_country_data(y_metric)
        elif y_metric == "growth_factor":
            by_country = self.get_country_data("confirmed")
        elif y_metric == "active":
            by_country = self.get_country_data("confirmed") - \
                         self.get_country_data("deaths") - \
                         self.get_country_data("recovered")
        elif len(ratio_parts) == 2 and ratio_parts[0] in self.dataframes and ratio_parts[1] in self.dataframes:
            by_country = self.get_country_data(ratio_parts[0]) / self.get_country_data(ratio_parts[1])
            by_country = by_country[self.get_country_data(ratio_parts[1]) > min_cases]
        else:
            print(f"{y_metric}' is an invalid y_metric!")

        for i, country in enumerate(countries_to_plot):
            if country not in by_country.index:
                raise KeyError(f"Country '{country}' not found!")
                return
            country_data = by_country.loc[country]  # , dates]
            fill = fills[i % (2 * m) < m]

            if fixed_country_colors:
                color = string_to_color(country)
            else:
                color = scalarMap.to_rgba(i)

            if y_metric == "growth_factor":
                df = self.get_new_cases_details(country, sigma)
                if x_metric == "day_number":
                    df = df[df.iloc[:, 0] >= min_cases]
                country_data = df.filtered_growth_factor
            is_valid = sum(np.nan_to_num(country_data)) > 0

            if x_metric == "calendar_date" and is_valid:
                plt.plot(country_data, marker=markers[i % m], label=country,
                         markersize=6, color=color, alpha=1, fillstyle=fill)

            elif x_metric == "day_number":
                if y_metric != "growth_factor":
                    country_data = country_data[country_data >= min_cases]
                if country == "Outside China":
                    length = len(country_data)
                day_nr = list(range(len(country_data)))
                if is_valid:
                    plt.plot(day_nr, country_data, marker=markers[i % m], label=country,
                             markersize=6, color=color, alpha=1, fillstyle=fill)

        if y_metric in short_metric_to_long:
            long_y_metric = short_metric_to_long[y_metric]
        else:
            long_y_metric = y_metric
        plt.ylabel(long_y_metric, fontsize=14)
        if x_metric == "calendar_date":
            plt.xlabel("Date", fontsize=14)
            plt.title(f"COVID-19 {long_y_metric} over time in selected countries", fontsize=18)
            plt.ylim(0.9 * use_log_scale,
                     by_country.loc[countries_to_plot].max().max() * (2 - 0.9 * (not use_log_scale)))
            firstweekday = pd.Timestamp(country_data.index[0]).dayofweek
            plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1, byweekday=firstweekday))
        elif x_metric == "day_number":
            if y_metric != "growth_factor":
                floor = 10 ** math.floor(math.log(min_cases) / math.log(10))
                floor = floor * (1 - (not use_log_scale)) * .9
                ceil = 10 ** math.ceil(math.log(by_country.loc[countries_to_plot].max().max()) / math.log(10))
                ceil = ceil * 1.2
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
        elif len(ratio_parts) > 1:
            plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{x:.1%}"))
            if use_log_scale:
                plt.yscale("log")
                plt.ylim((0.001, 0.12))
            else:
                plt.ylim((0, 0.12))
        else:
            set_y_axis_format(country_data, use_log_scale)
        plt.grid()
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.gca().tick_params(which="minor", width=0)
        plt.gca().tick_params(which="major", color=light_grey)
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.show()

    def plot_pie(self, y_metric):
        short_y = y_metric.split()[0]
        plt.figure(figsize=(8, 8))
        data_for_pie = self.dataframes[short_y + "_by_country"].iloc[:, -1]
        data_for_pie = data_for_pie[data_for_pie.index != "All except China"]
        data_for_pie = data_for_pie.sort_values(ascending=False)
        countrynames = [x if data_for_pie[x] / data_for_pie.values.sum() > .015 else "" for x in data_for_pie.index]
        data_for_pie.plot.pie(startangle=270, autopct=get_pie_label, labels=countrynames,
                              counterclock=False, pctdistance=.75,
                              colors=[string_to_color(x) for x in data_for_pie.index])

        plt.ylabel("")
        plt.title(f"{y_metric.capitalize()} as of {data_for_pie.name.date()}", fontsize=16)
        plt.show()

    def curve_fit(self, country="All except China", days=100, do_plot=True):
        x = np.arange(days)
        country_data = self.dataframes["confirmed_by_country"].loc[country, :]
        country_data = country_data[np.isfinite(country_data)]
        current_day = country_data.index[-1]

        [L, k, x0], pcov = scipy.optimize.curve_fit(logistic_func, np.arange(len(country_data)),
                                                    country_data, maxfev=10000,
                                                    p0=[country_data.max()*2, 0.5, np.clip(len(country_data), 1, 200)],
                                                    bounds=([1, 0.1, 1], [1e9, 0.999, 400]),
                                                    method="trf"
                                                    )

        # dates up to 100 days after start
        model_date_list = [current_day + datetime.timedelta(days=n) for n in range(0, len(x) - len(country_data))]
        model_date_list = [mdates.date2num(x) for x in model_date_list]

        n = len(model_date_list)
        sig_L, sig_k, sig_x0 = np.sqrt(np.diag(pcov))
        logistic = logistic_func(x[-n:] - 1, L, k, x0)
        logistic_sigma = logistic_func(x[-n:], sig_L, sig_k, sig_x0)
        uncertainty_w = np.linspace(0, 1, len(logistic))

        if do_plot:
            plt.plot(country_data, label="Confirmed cases in " + country, markersize=3, zorder=1)
            plt.plot(model_date_list, np.round(logistic), 
                     label=f"{L:.0f} / (1 + e^(-{k:.3f} * (x - {x0:.1f})))", zorder=1)

            plt.grid()
            plt.legend(loc="upper left")
            plt.title(f"Logistic curve fit and extrapolation for {country}", fontsize=18)
            plt.xlabel("Date", fontsize=14)
            plt.ylabel("Cases", fontsize=14)
            plt.scatter(mdates.date2num(current_day), country_data[-1], s=20, c="C00", zorder=2)
            plt.annotate(
                f"{datetime.datetime.strftime(current_day, '%m/%d')}: {kmb_number_format(country_data[-1], 3, 0)}",
                (mdates.date2num(current_day) - 1, country_data[-1]), fontsize=18, ha="right")

            plt.scatter(model_date_list[-1], logistic[-1], s=20, c="C01", zorder=2)
            plt.annotate(
                f"{mdates.num2date(model_date_list[-1]).strftime('%m/%d')}: {kmb_number_format(logistic[-1], 3, 0)}",
                         (model_date_list[-1] - 1, logistic[-1] * 1.08), fontsize=18, ha="right")
            set_y_axis_format(logistic, True)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.gca().tick_params(which="both", color=light_grey)
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            bottom, top = plt.ylim()
            plt.ylim((bottom, max(bottom+1, top)))
            plt.show()

    def simulate_country_history(self, country, population, history_length=28, show_result=False):
        confirmed = self.dataframes["confirmed_by_country"].loc[country]
        deaths = self.dataframes["deaths_by_country"].loc[country]
        recovered = self.dataframes["recovered_by_country"].loc[country]
        active = confirmed - deaths - recovered
        uninfected = (population - confirmed).fillna(population)
        simulation = pd.DataFrame(data=[confirmed, deaths, recovered, active, uninfected],
                                  index=["confirmed", "deaths", "recovered", "active", "uninfected"]).transpose()
        simulation = simulation.fillna(0)

        # reconstruct active case durations by assuming FIFO
        for i, day in enumerate(confirmed.index):
            case_history = [0] * history_length
            if i == 0:
                case_history[0] = simulation.confirmed[0]
            else:
                new_cases = simulation.confirmed.diff()[i]
                new_deaths = simulation.deaths.diff()[i]
                new_recovered = simulation.recovered.diff()[i]
                newly_resolved = new_deaths + new_recovered

                case_history[0] = new_cases
                case_history[1:] = simulation.iloc[i - 1, -history_length:]
                case_history = case_history[:history_length]

            for h in range(history_length):
                x = history_length - h - 1
                # print(h, x, case_history)
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
            self,
            country,  # name of the country to simulate
            population,  # population of the country
            days=30,  # how many days into the future to simulate
            cfr=0.02,  # case fatality rate, 0 to 1
            history_length=21,  # length of case history
            sigma_death_days=5,  # Standard deviation in mortality over time distribution
            growth_rate_trend=[1.2, 0.8]  # Growth factor development over time. This will be linearly
            # interpolated to a vector of length {days}
    ):
        growth_rate_per_day = np.interp(np.linspace(0, 1, days),
                                        np.linspace(0, 1, len(growth_rate_trend)),
                                        growth_rate_trend)

        country_history = self.simulate_country_history(country, population, history_length)
        daily_death_chance = death_chance_per_day(cfr, sigma_death_days, history_length, do_plot=False)
        today = country_history.index[-1]

        for d in range(days):
            # column shortcuts
            confirmed = country_history.confirmed
            deaths = country_history.deaths
            active = country_history.active
            recovered = country_history.recovered
            alive = population - deaths.iloc[-1]
            uninfected = int(np.maximum(0, population - confirmed.iloc[-1]))
            case_history = country_history.iloc[-1, -history_length:].copy()

            last_day = confirmed.index[-1]
            next_day = last_day + pd.DateOffset(1)
            daily_growth = growth_rate_per_day[d]
            last_delta = confirmed[-1] - confirmed[-2]

            # Infect
            # TODO: Use R0 and base new cases on active cases, rather than 
            #       growth factor and new cases based on new cases the day before
            last_confirmed = confirmed.loc[last_day]
            uninfected_ratio = uninfected / alive
            new_cases = int(np.maximum(0, last_delta * daily_growth * uninfected_ratio))

            # Deaths
            new_deaths = 0
            for case_duration in range(history_length):
                deaths_for_duration = np.random.binomial(case_history[case_duration],
                                                         daily_death_chance[case_duration])
                case_history[case_duration] -= deaths_for_duration
                new_deaths += deaths_for_duration

            # Recoveries
            new_recovered = case_history[-1]

            # Shift case history
            case_history[1:] = case_history[:-1]
            case_history.iloc[0] = new_cases

            country_history.at[next_day, "confirmed"] = last_confirmed + new_cases
            country_history.at[next_day, "deaths"] = deaths.loc[last_day] + new_deaths
            country_history.at[next_day, "recovered"] = recovered.loc[last_day] + new_recovered
            country_history.at[next_day, "active"] = case_history.sum()
            country_history.at[next_day, "uninfected"] = uninfected
            country_history.iloc[-1, -history_length:] = case_history

        return country_history, today

    def plot_simulation(self, country, population, days, growth_rate_trend,
                        history_length=21, do_log=False, scenario_name=""):
        simulation, today = self.simulate_country(country=country, population=population, days=days,
                                                  growth_rate_trend=growth_rate_trend, history_length=history_length)

        plt.figure(figsize=(13, 8))
        metrics = ["confirmed cases", "deaths", "active cases", "recovered cases"]
        cm = plt.cm.get_cmap("tab10")

        for i, metric in enumerate(metrics):
            short_metric = metric.split()[0]
            plt.plot(simulation.loc[:today, short_metric], c=cm.colors[i], label=f"{metric.capitalize()}")
            plt.plot(simulation.loc[today:, short_metric], "-.", c=cm.colors[i], alpha=0.75)
            plt.grid()
            plt.legend(loc="upper left")

        set_y_axis_format(simulation.loc[:, "confirmed"], log=do_log)
        title = f"Covid-19 simulation for {country} for the next {days} days"
        if scenario_name:
            title += ": " + scenario_name + " scenario"
        plt.suptitle(title, fontsize=20, y=1.03)
        plt.tight_layout()
        plt.grid()
        plt.show()
        simulation = simulation.astype(int)
        display(Markdown(f"### {scenario_name} final tally:"))
        peak_active = simulation.active.max()
        peak_active_date = simulation.active[simulation.active == simulation.active.max()].index[0].date()
        print(f"Confirmed: {kmb_number_format(simulation.confirmed[-1], 3 , False)},\n" +
              f"Deaths: {kmb_number_format(simulation.deaths[-1], 3 , False)},\n" +
              f"Recovered: {kmb_number_format(simulation.recovered[-1], 3 , False)},\n"
              f"Peak active: {kmb_number_format(peak_active, 3, False)} at {peak_active_date}")

        return simulation

    def country_highlight(self, country):
        metrics = ["new_cases", "new_deaths"]
        country_data = self.get_new_cases_details(country).round(2)[metrics]
        display(country_data.tail(7))

        for metric in metrics:
            data = country_data[metric]
            plt.plot(country_data.index, data, label=metric.capitalize().replace("_", " "))

        plt.title(f"{country} situation as of {country_data.index[-1].date()}", fontsize=20)
        plt.grid()
        plt.legend()
        set_y_axis_format(country_data[metrics], log=True)
        plt.show()
