from covid19_util import *

from matplotlib import dates as mdates
from matplotlib import colors as mcolors
import pandas as pd
import requests
import scipy.optimize
import scipy.stats
from io import StringIO
import datetime
import geonamescache


class Covid19Processing:
    def __init__(self):
        self.dataframes = {}
        gc = geonamescache.GeonamesCache()
        gc_data = gc.get_countries()
        self.country_metadata = {}
        normalized_names = {
            "Timor Leste": "East Timor",
            "Vatican": "Vatican City",
            "Democratic Republic of the Congo": "Congo (Kinshasa)",
            "Republic of the Congo": "Congo (Brazzaville)",
            "Cabo Verde": "Cape Verde"
        }

        for country_code in gc_data:
            metadata = gc_data[country_code]
            name = metadata["name"]
            if name in normalized_names:
                name = normalized_names[name]
            population = metadata["population"]
            area = metadata["areakm2"]
            continent = continent_codes[metadata["continentcode"]]

            self.country_metadata[name] = {
                "population": population,
                "area": area,
                "continent": continent
            }

        for metric in data_urls.keys():
            url = base_url + data_urls[metric]  # Combine URL parts
            r = requests.get(url)  # Retrieve from URL
            self.dataframes[metric] = pd.read_csv(StringIO(r.text), sep=",")  # Convert into Pandas dataframe

        # Display the first lines
        display(Markdown("### Raw confirmed cases data, per region/state"))
        with pd.option_context("display.max_rows", 10, "display.max_columns", 14):
            display(self.dataframes["confirmed"])

    def process(self, rows=20, debug=False):
        # Clean up
        for metric in data_urls.keys():
            by_country = self.dataframes[metric].groupby("Country/Region").sum()  # Group by country
            dates = by_country.columns[2:]  # Drop Lat/Long

            # Convert to columns to matplotlib dates
            by_country = by_country.loc[:, dates]
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

                # Correct a typo in US data, see https://github.com/CSSEGISandData/COVID-19/issues/2167
                if by_country.loc["US", pd.to_datetime("4/13/20")] == 682619:
                    by_country.loc["US", pd.to_datetime("4/13/20")] -= 102000

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
                                                  "occupied Palestinian territory": "Palestine",
                                                  "West Bank and Gaza": "Palestine",
                                                  "Bahamas, The": "Bahamas",
                                                  "Cote d'Ivoire": "Ivory Coast",
                                                  "Gambia, The": "Gambia",
                                                  "US": "United States",
                                                  "Cabo Verde": "Cape Verde",
                                                  })
            by_country.sort_index(inplace=True)

            # Store processed results for metric
            self.dataframes[metric + "_by_country"] = by_country.fillna(0).astype(int)

        # Add in recovered and active
        self.dataframes["recovered_by_country"] = pd.DataFrame(columns=self.dataframes["confirmed_by_country"].columns)
        self.dataframes["active_by_country"] = pd.DataFrame(columns=self.dataframes["confirmed_by_country"].columns)
        for country in self.dataframes["confirmed_by_country"].index:
            simulation = self.simulate_country_history(country, history_length=40)
            self.dataframes["recovered_by_country"].loc[country, :] = simulation.recovered
            self.dataframes["active_by_country"].loc[country, :] = simulation.active

        # Add in continents
        for metric in list(data_urls.keys()) + ["recovered", "active"]:
            continent_data = {}
            by_country = self.dataframes[metric+"_by_country"]
            for country in by_country.index:
                if country in self.country_metadata:
                    continent = self.country_metadata[country]["continent"]
                    if continent in continent_data:
                        continent_data[continent] += by_country.loc[country, :]
                    else:
                        continent_data[continent] = by_country.loc[country, :]

                elif metric == "confirmed" and debug:
                    print(f"Missing metadata for {country}!")

            by_continent = pd.DataFrame(columns=by_country.columns)
            for continent in continent_data:
                by_continent.loc[continent, :] = continent_data[continent]

            # Add in special regions
            all_countries = by_country.sum()
            by_continent.loc["All except China", :] = all_countries - by_country.loc["China", dates]
            by_continent.loc["World", :] = all_countries
            by_continent = by_continent
            self.dataframes[metric + "_by_continent"] = by_continent.fillna(0).astype(int)

        # Add population for special regions and continents
        continent_pop = {}
        for country in by_country.index:
            if country in self.country_metadata:
                continent = self.country_metadata[country]["continent"]
                pop = self.country_metadata[country]["population"]
                if continent in continent_pop:
                    continent_pop[continent] += pop
                else:
                    continent_pop[continent] = pop
        continent_pop["World"] = sum(continent_pop.values())
        continent_pop["All except China"] = continent_pop["World"] - self.country_metadata["China"]["population"]
        for c in continent_pop:
            if c not in self.country_metadata:
                self.country_metadata[c] = {}
            self.country_metadata[c]["population"] = continent_pop[c]

        with pd.option_context("display.max_rows", rows, "display.min_rows", rows, "display.max_columns", 10):
            display(Markdown("### Table of confirmed cases by country"))
            display(self.dataframes["confirmed_by_country"])
            display(Markdown("### Table of confirmed cases by continent/region"))
            display(self.dataframes["confirmed_by_continent"])

    def list_countries(self, columns=5):
        confirmed_by_country = self.dataframes["confirmed_by_country"]
        n_countries = len(confirmed_by_country)
        display(Markdown(f"### {n_countries} countries/territories affected:\n"))
        for i, k in enumerate(confirmed_by_country.index):
            if len(k) > 19:
                k = k[:18].strip() + "."
            print(f"{k:20}", end=" " if (i + 1) % columns else "\n")  # Every 5 items, end with a newline

    def get_metric_data(self, metric):
        if metric+"_by_country" in self.dataframes:
            return pd.concat([self.dataframes[metric + "_by_country"], self.dataframes[metric + "_by_continent"]])
        elif metric.startswith("new") and metric.split(" ")[1] in self.dataframes:
            metric = metric.split(" ")[1]
            combined = pd.concat(
                [self.dataframes[metric + "_by_country"].diff(axis="columns"),
                 self.dataframes[metric + "_by_continent"].diff(axis="columns")]
            )
            combined[combined < 1] = np.nan
            return combined
        elif metric in self.country_metadata["China"]:
            all_regions = self.dataframes["confirmed_by_country"].index.tolist() +\
                           self.dataframes["confirmed_by_continent"].index.tolist()
            s = pd.Series(
                {country: self.country_metadata[country][metric] for country in self.country_metadata}, name=metric)
            s = s[s.index.isin(all_regions)]  # Remove countries not in JHU data
            return s.round()
        else:
            return

    def get_country_data(self, country):
        data = {}
        for metric in self.dataframes.keys():
            if not metric.endswith("by_country"):
                continue
            series = self.dataframes[metric].loc[country, :]
            series.name = metric
            data[metric] = series
        return pd.DataFrame(data)

    def get_new_cases_details(self, country, avg_n=5, median_n=3):
        confirmed = self.get_metric_data("confirmed").loc[country]
        deaths = self.get_metric_data("deaths").loc[country]
        df = pd.DataFrame(confirmed)
        df = df.rename(columns={country: "confirmed_cases"})
        df.loc[:, "new_cases"] = np.maximum(0, confirmed.diff())
        df.loc[:, "new_deaths"] = np.maximum(0, deaths.diff())
        df = df.fillna(0)
        df.loc[:, "growth_factor"] = df.new_cases.diff() / df.new_cases.shift(1) + 1
        df[~np.isfinite(df)] = np.nan
        df.loc[:, "filtered_new_cases"] = \
            scipy.ndimage.convolve(df.new_cases, np.ones(avg_n) / avg_n, origin=-avg_n // 2 + 1)
        df.loc[:, "filtered_growth_factor"] = \
            df.filtered_new_cases.diff() / df.filtered_new_cases.shift(1) + 1
        df.filtered_growth_factor = scipy.ndimage.median_filter(df.filtered_growth_factor, median_n, mode="nearest")
        return df

    def plot(self, x_metric, y_metric, countries_to_plot, colormap=cm, use_log_scale=True,
             min_cases=0, sigma=5, fixed_country_colors=False):

        # layout/style stuff
        markers = ["o", "^", "v", "<", ">", "s", "X", "D", "*", "$Y$", "$Z$"]
        short_metric_to_long = {
            "confirmed": "Confirmed cases",
            "deaths": "Deaths",
            "active": "Active cases",
            "growth_factor": f"{sigma}-day avg growth factor",
            "deaths/confirmed": "Case fatality",
            "new confirmed": "Daily new cases",
            "confirmed/population": "Confirmed cases per million population",
            "active/population": "Active cases per million population",
            "deaths/population": "Deaths per million population"
        }
        fills = ["none", "full"]  # alternate between filled and empty markers
        length = None
        m = len(markers)
        cm = plt.cm.get_cmap(colormap)
        n_colors = min(len(markers), len(countries_to_plot))
        c_norm = matplotlib.colors.Normalize(vmin=0, vmax=n_colors)
        scalar_map = matplotlib.cm.ScalarMappable(norm=c_norm, cmap=cm)
        y_max = 0
        ratio_parts = y_metric.split("/")

        if self.get_metric_data(y_metric) is not None:
            by_country = self.get_metric_data(y_metric)
        elif y_metric == "growth_factor":
            by_country = self.get_metric_data("confirmed")
        elif y_metric == "active":
            by_country = self.get_metric_data("confirmed") - \
                         self.get_metric_data("deaths") - \
                         self.get_metric_data("recovered")
            by_country = by_country.dropna("columns").astype(int)
        elif len(ratio_parts) == 2 and self.get_metric_data(ratio_parts[0]) is not None\
                and self.get_metric_data(ratio_parts[1]) is not None:
            numerator = self.get_metric_data(ratio_parts[0])
            denominator = self.get_metric_data(ratio_parts[1])
            numerator = numerator.loc[denominator.index, :]
            if ratio_parts[1] != "population":
                numerator = numerator[denominator > min_cases]
                denominator = denominator[denominator > min_cases]

            by_country = numerator.divide(denominator, 0)  # numerator / denominator
            if ratio_parts[1] == "population":
                by_country *= 1e6

            if use_log_scale:
                by_country[by_country == 0] = np.nan
        else:
            print(f"'{y_metric}' is an invalid y_metric!")

        if len(by_country) >= 20:
            n = len(by_country) // 20
            mark_every = slice(-1, 0, -n)
        else:
            mark_every = None

        for i, country in enumerate(countries_to_plot):
            if country in by_country.index:
                country_data = by_country.loc[country].dropna()
            if country not in by_country.index:
                raise KeyError(f"Country '{country}' not found for {y_metric}!")
                return

            marker_fill = fills[i % (2 * m) < m]

            if fixed_country_colors:
                color = string_to_color(country)
            else:
                color = scalar_map.to_rgba(i % n_colors)
                hsv = mcolors.rgb_to_hsv(color[:3])
                if 0.12 <= hsv[0] < 0.25:
                    hsv[0] -= 0.02
                    hsv[2] = min(hsv[2] - 0.02, 1)
                    color = mcolors.hsv_to_rgb(hsv)

            if y_metric == "growth_factor":
                df = self.get_new_cases_details(country, sigma)
                if x_metric == "day_number":
                    df = df[df.iloc[:, 0] >= min_cases]
                country_data = df.filtered_growth_factor

            if x_metric == "calendar_date":
                x_data = country_data.index

            elif x_metric == "day_number":
                if y_metric != "growth_factor" and len(ratio_parts) < 2:
                    country_data = country_data[country_data >= min_cases]
                if country == "Outside China":
                    length = len(country_data)
                x_data = list(range(len(country_data)))

            plt.plot(x_data, country_data, marker=markers[i % m], label=country, markevery=mark_every,
                     markersize=7, color=color, alpha=0.8, fillstyle=marker_fill)

            if country_data.max() is not np.nan:
                mx = country_data.max()
                if not np.isscalar(mx):
                    m = m.max()
                y_max = max(y_max, mx)

        if y_metric in short_metric_to_long:
            long_y_metric = short_metric_to_long[y_metric]
        else:
            long_y_metric = y_metric
        plt.ylabel(long_y_metric, fontsize=14)
        if x_metric == "calendar_date":
            plt.xlabel("Date", fontsize=14)
            title = f"COVID-19 {long_y_metric}"
            plt.title(title, fontsize=18)
            if not ratio_parts[-1] == "population":
                plt.ylim(0.9 * use_log_scale,
                         by_country.loc[countries_to_plot].max().max() * (2 - 0.9 * (not use_log_scale)))
            firstweekday = pd.Timestamp(by_country.iloc[0].index[0]).dayofweek
            n_days = (country_data.index.max() - country_data.index.min()).days + 1
            n_weeks = n_days//5
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
            plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=n_weeks//7, byweekday=firstweekday))
        elif x_metric == "day_number":
            if y_metric != "growth_factor" and len(ratio_parts) < 2:
                floor = 10 ** math.floor(math.log(min_cases) / math.log(10))
                floor = floor * (1 - (not use_log_scale)) * .9
                ceil = 10 ** math.ceil(math.log(by_country.loc[countries_to_plot].max().max()) / math.log(10))
                ceil = ceil * 1.2
                plt.ylim(floor, ceil)
            plt.xlim(0, length)
            plt.xlabel("Day Number", fontsize=14)
            if len(ratio_parts) < 2:
                title = f"COVID-19 {long_y_metric}, from the first day with ≥{min_cases} cases"
            else:
                title = f"COVID-19 {long_y_metric} ratio in selected countries"
            plt.title(title, fontsize=18)

        plt.legend(frameon=False)
        if y_metric == "growth_factor":
            plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{x:,.2f}"))
        elif ratio_parts[-1] == "population":
            pass
        elif len(ratio_parts) > 1:
            plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{x:.1%}"))
            if use_log_scale:
                plt.yscale("log")
                plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{x:.3f}"))
            else:
                pass
        else:
            set_y_axis_format(y_max, use_log_scale)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.gca().tick_params(which="major", color=light_grey)
        set_plot_style()
        plt.show()

    def plot_pie(self, y_metrics, mode="country"):
        plt.figure(figsize=(8, 8*len(y_metrics)))
        for i, y_metric in enumerate(y_metrics):
            plt.subplot(len(y_metrics), 1, i+1)
            short_y = y_metric.split()[0]
            data_for_pie = self.dataframes[short_y + "_by_"+mode].iloc[:, -1]
            data_for_pie = data_for_pie[~data_for_pie.index.isin(["All except China", "World"])]
            data_for_pie = data_for_pie.sort_values(ascending=False).fillna(0)
            data_for_pie = np.maximum(0, data_for_pie)
            country_names = [x if data_for_pie[x] / data_for_pie.sum() > .015 else "" for x in data_for_pie.index]
            data_for_pie.plot.pie(startangle=270, autopct=get_pie_label, labels=country_names,
                                  counterclock=False, pctdistance=.75,
                                  colors=[string_to_color(x) for x in data_for_pie.index],
                                  textprops={'fontsize': 12})

            plt.ylabel("")
            plt.title(f"{y_metric.capitalize()} as of {data_for_pie.name.date()}", fontsize=16)
        plt.show()

    def curve_fit(self, country="All except China", days=100, do_plot=True):
        country_data = self.get_metric_data("confirmed").loc[country, :]
        country_data = country_data[np.isfinite(country_data)]
        x = np.arange(len(country_data), days + len(country_data))
        current_day = country_data.index[-1]
        if country in self.country_metadata:
            population = self.country_metadata[country]["population"]
        else:
            population = 1e8

        [L, k, x0], pcov = scipy.optimize.curve_fit(logistic_func, np.arange(len(country_data)),
                                                    country_data, maxfev=10000,
                                                    p0=[country_data[-1], 0.5, np.clip(len(country_data), 1, 365)],
                                                    bounds=([0, 0, 0], [population, 1, 800]),
                                                    method="trf"
                                                    )

        # dates up to 'days' days after present
        model_date_list = [country_data.index[-1] + datetime.timedelta(days=n) for n in range(days)]
        model_date_list = [mdates.date2num(x) for x in model_date_list]

        n = len(model_date_list)
        logistic = logistic_func(x - 2, L, k, x0)

        if do_plot:
            plt.plot(country_data, label="Confirmed cases in " + country, markersize=3, zorder=1)
            plt.plot(model_date_list, np.round(logistic),
                     label=f"{L:.0f} / (1 + e^(-{k:.3f} * (x - {x0:.1f})))", zorder=1)

            plt.legend(loc="lower right")
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
            set_y_axis_format(logistic.max(), True)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.gca().tick_params(which="both", color=light_grey)
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            bottom, top = plt.ylim()
            plt.ylim((bottom, max(bottom+1, top)))
            set_plot_style()
            plt.show()

    def simulate_country_history(self, country, history_length=40, show_result=False):
        if country in self.country_metadata:
            population = self.country_metadata[country]["population"]
        else:
            population = np.nan
        confirmed = self.dataframes["confirmed_by_country"].loc[country]
        deaths = self.dataframes["deaths_by_country"].loc[country]
        recovered = np.zeros(len(confirmed))
        active = np.zeros(len(confirmed))
        uninfected = (population - confirmed).fillna(population)
        simulation = pd.DataFrame(data=[confirmed, deaths, recovered, active, uninfected],
                                  index=["confirmed", "deaths", "recovered", "active", "uninfected"]).transpose()
        simulation = simulation.fillna(0)
        daily_death_distribution = death_chance_per_day(cfr=0.04, s=1.75, mu=0.5, sigma=10, length=history_length)

        # reconstruct recovered and active case durations using fatality by case duration stats
        for i, day in enumerate(confirmed.index):
            case_history = np.zeros(history_length)
            if i == 0:
                new_recovered = 0
            else:
                new_cases = simulation.confirmed.diff()[i]
                new_deaths = simulation.deaths.diff()[i]
                new_deaths_by_case_duration = (new_deaths * daily_death_distribution)

                # insert new cases for the day
                case_history[0] = new_cases

                # shift previous cases
                case_history[1:] = simulation.iloc[i - 1, -history_length:-1]
                case_history = case_history[:history_length]

                # subtract deaths
                case_history -= new_deaths_by_case_duration

                # counteract difference between theoretical mortality distribution and reality
                case_history = np.maximum(0, case_history)
                new_recovered = simulation.recovered.iloc[i - 1] + max(0, case_history[-1])

            for h in range(history_length):
                simulation.at[day.to_datetime64(), f"active_{h}"] = case_history[h]
            simulation.at[day.to_datetime64(), f"recovered"] = new_recovered
            simulation.at[day.to_datetime64(), f"active"] = sum(case_history)

        simulation = simulation.fillna(0).astype(int)
        if show_result:
            display(Markdown(f"<br>**Last 10 days in {country}, showing a 7-day case duration history:**"))
            display(simulation.iloc[-10:, :])
        return simulation

    def simulate_country(
            self,
            country,                 # name of the country to simulate
            days=30,                 # how many days into the future to simulate
            cfr=0.03,                # case fatality rate, 0 to 1
            critical_rate=0.18,      # https://jamanetwork.com/journals/jama/fullarticle/2763188, https://www.cdc.gov/mmwr/volumes/69/wr/mm6912e2.htm
            cfr_without_icu=0.80,    # unknown but high
            icu_beds_per_100k=30,    # https://www.forbes.com/sites/niallmccarthy/2020/03/12/the-countries-with-the-most-critical-care-beds-per-capita-infographic
            icu_availability=0.2,    #
            history_length=28,       # Length of case history
            sigma_death_days=6,      # Standard deviation in mortality over time distribution
            r0=2.5,
            mitigation_start=1.0,    # Initial mitigation factor
            mitigation_end=1.0,      # Final mitigation factor. This will be linearly interpolated to a trend vector
            from_day=-1
    ):
        population = self.country_metadata[country]["population"]
        country_history = self.simulate_country_history(country, history_length)
        today = country_history.index[-1]
        if from_day != -1:
            days = (today - country_history.index[from_day]).days
            country_history = country_history[country_history.confirmed > 0]
            country_history = country_history.iloc[:from_day+1, :]
        available_icu_beds = int(population/100000 * icu_beds_per_100k * icu_availability)

        # effective mitigation ramps up to given mitigation factor over 14 days
        daily_mitigation = np.append(np.linspace(mitigation_start, mitigation_end, 14), max(0, days - 14) * [mitigation_end])

        daily_death_chance = death_chance_per_day(cfr, 1.75, 0.5, sigma_death_days, history_length, do_plot=False)
        #  daily_death_chance_no_icu = death_chance_per_day(cfr_without_icu, 1.75, 0.5,
        #                                                sigma_death_days, history_length, do_plot=False)

        # https://www.jwatch.org/na51083/2020/03/13/covid-19-incubation-period-update
        # https://www.medrxiv.org/content/10.1101/2020.03.05.20030502v1.full.pdf
        daily_transmission_chance = scipy.stats.norm.pdf(np.linspace(0, history_length, history_length+1),
                                                         loc=4.5, scale=1.6)

        for d in range(days):
            # column shortcuts
            confirmed = country_history.confirmed
            deaths = country_history.deaths
            recovered = country_history.recovered
            case_history = country_history.iloc[-1, -history_length:].copy()

            last_day = confirmed.index[-1]
            next_day = last_day + pd.DateOffset(1)

            current_alive = population - deaths.iloc[-1]
            current_uninfected = int(np.maximum(0, population - confirmed.iloc[-1]))
            current_uninfected_ratio = current_uninfected / current_alive
            current_mitigation = daily_mitigation[d]

            # Infect
            r_eff = r0 * current_mitigation * current_uninfected_ratio
            new_cases = 0
            for case_duration in range(history_length):
                new_cases_for_case_duration = np.random.binomial(case_history[case_duration],
                                                                 r_eff*daily_transmission_chance[case_duration])
                new_cases += int(round(new_cases_for_case_duration))

            # Deaths
            new_deaths = 0
            for case_duration in range(history_length):
                cases = case_history[case_duration]
                # TODO: assign patients to mild or critical only once
                # critical_patients = (critical_rate * cases).round()
                # critical_patients_in_icu = min(available_icu_beds, critical_patients)
                # critical_patients_no_icu = max(0, critical_patients - available_icu_beds)
                non_icu_patients = cases  # - critical_patients

                deaths_for_case_duration = np.random.binomial(non_icu_patients,  # + critical_patients_in_icu,
                                                              daily_death_chance[case_duration])

                # deaths_for_case_duration += np.random.binomial(critical_patients_no_icu,
                #                                              daily_death_chance_no_icu[case_duration])

                case_history[case_duration] -= deaths_for_case_duration
                new_deaths += deaths_for_case_duration

            # Recoveries
            new_recovered = case_history[-1]

            # Shift case history
            case_history[1:] = case_history[:-1]
            case_history.iloc[0] = new_cases

            country_history.at[next_day, "confirmed"] = confirmed.loc[last_day] + new_cases
            country_history.at[next_day, "deaths"] = deaths.loc[last_day] + new_deaths
            country_history.at[next_day, "recovered"] = recovered.loc[last_day] + new_recovered
            country_history.at[next_day, "active"] = case_history.sum()
            country_history.at[next_day, "uninfected"] = current_uninfected
            country_history.iloc[-1, -history_length:] = case_history

        return country_history, today

    def plot_simulation(self, country, days, mitigation_trend, cfr=0.02, r0=2.5,
                        history_length=30, use_log_scale=True, scenario_name="", from_day=-1):

        simulation, today = self.simulate_country(country=country, days=days, cfr=cfr,
                                                  mitigation_start=mitigation_trend[0],
                                                  mitigation_end=mitigation_trend[1],
                                                  r0=r0,
                                                  history_length=history_length,
                                                  from_day=from_day)

        plt.figure(figsize=(13, 8))
        metrics = ["confirmed cases", "deaths", "active cases", "recovered cases"]
        c = ["tab:blue", "r", "tab:orange", "limegreen", "tab:purple"]

        for i, metric in enumerate(metrics):
            short_metric = metric.split()[0]
            plt.plot(simulation.loc[:today, short_metric], c=c[i], label=f"{metric.capitalize()}")
            plt.plot(simulation.loc[today:, short_metric], "-.", c=c[i], alpha=0.75)
        plt.plot(simulation.loc[today - pd.DateOffset(1):, "confirmed"].diff(), "-.", c=c[i + 1], alpha=0.75)
        plt.plot(simulation.loc[:today, "confirmed"].diff(), c=c[-1], label="Daily new cases")
        plt.legend(loc="upper left")

        set_y_axis_format(simulation.loc[:, "confirmed"].max().max(), log=use_log_scale)
        title = f"{days}-day Covid-19 simulation, {country}"
        if scenario_name:
            title += ": " + scenario_name
        plt.suptitle(title, fontsize=20, y=1.03)
        plt.tight_layout()
        set_plot_style()
        plt.show()
        simulation = simulation.astype(int)
        display(Markdown(f"### {scenario_name} final tally:"))
        peak_active = simulation.active.max()
        peak_active_date = simulation.active[simulation.active == simulation.active.max()].index[0].date()
        print(f"Confirmed: {kmb_number_format(simulation.confirmed[-1], 3 , False)},\n" 
              f"Deaths: {kmb_number_format(simulation.deaths[-1], 3 , False)},\n" 
              f"Recovered: {kmb_number_format(simulation.recovered[-1], 3 , False)},\n"
              f"Peak active: {kmb_number_format(peak_active, 3, False)} at {peak_active_date},\n"
              f"Uninfected: {kmb_number_format(simulation.uninfected[-1], 3 , False)}"
              )

        return simulation

    def country_highlight(self, country):
        metrics = ["new_cases", "new_deaths"]
        country_data = self.get_new_cases_details(country).round(2)[metrics]
        display(country_data.tail(7).astype(int))

        for metric in metrics:
            data = country_data[metric]
            plt.plot(country_data.index, data, label=metric.capitalize().replace("_", " "))

        plt.title(f"{country} daily changes as of {country_data.index[-1].date()}", fontsize=20)
        set_plot_style()
        plt.legend(loc="upper left")
        set_y_axis_format(country_data[metrics].max().max(), log=True)
        plt.show()
