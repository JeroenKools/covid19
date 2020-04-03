# currently unused

from covid19_util import *

from scipy import stats
import numpy as np


def weibull(length, shape, mu, sigma):
    return stats.weibull_min.pdf(np.linspace(0, length-1, length), c=shape, loc=mu, scale=sigma)


def lognorm(length, shape, mu, sigma):
    return stats.lognorm(np.linspace(0, length-1, length), s=shape, loc=mu, scale=sigma)


def norm(length, mu, sigma):
    return scipy.stats.norm.pdf(np.linspace(0, length-1, length), loc=mu, scale=sigma)


def normalize_distribution(dist_type, magnitude, s=0.9, mu=0, sigma=1, length=20, do_plot=False):
    # Initial approximation)
    prob = dist_type(length, s, mu, sigma)

    # Approximation is slightly off, compensate
    if magnitude > 0:
        prob = prob / sum(prob) * magnitude
        complement = 1 - prob
        complement = normalize_to_target_product(complement, 1 - magnitude)
        prob = 1 - complement

    return prob


class Scenario:
    def __init__(self, cfr, r0, mitigation_trend, days_to_simulate, case_history_length,
                 death_distribution_type, death_distribution_params,
                 transmission_distribution_type, transmission_distribution_params
                 ):
        self.cfr = cfr
        self.r0 = r0
        self.days_to_simulate = days_to_simulate
        self.case_history_length = case_history_length
        self.death_distribution = death_distribution_type(*death_distribution_params)
        self.daily_mitigation = np.interp(np.linspace(0, 1, days_to_simulate),
                                          np.linspace(0, 1, len(mitigation_trend)),
                                          mitigation_trend
                                          ),
        self.transmission_distribution = transmission_distribution_type(*transmission_distribution_params)
        self.daily_transmission_chance = norm(*transmission_distribution_params)


class Simulation:
    def __init__(self, data, country):
        self.data = data
        self.country = country
        self.population = data.country_metadata[country]["population"]

    def generate_history(self):
        pass

    def run_scenario(self,
                     scenario_name, days_to_simulate=180, case_history_length=None,
                     cfr=0.03, r0=2.5, mitigation_trend=(1., 1.),
                     death_distribution_type=weibull,
                     death_distribution_params=(1.75, 0.5, 10),
                     transmission_distribution_type=norm,
                     transmission_distribution_params=(5.5, 1.6)):

        death_distribution = death_distribution_type(case_history_length, *death_distribution_params)
        daily_mitigation = np.interp(np.linspace(0, 1, days_to_simulate),
                                     np.linspace(0, 1, len(mitigation_trend)),
                                     mitigation_trend)
        transmission_distribution = transmission_distribution_type(case_history_length, *transmission_distribution_params)
        daily_transmission_chance = norm(*transmission_distribution_params)

