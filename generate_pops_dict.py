import pandas as pd
import geocoder
import geonamescache
import os

"""
Generates a dataframe of country/state populations
Requires a free geocoder account (https://geocoder.readthedocs.io/), and the 'geocoder_key'
env variable to be set to your api key (which is equal to your username).
"""

gnc = geonamescache.GeonamesCache()
states = [x["name"] for x in gnc.get_us_states().values()]
countries = [x["name"] for x in gnc.get_countries().values()]
populations = {}
key = os.environ["geocoder_key"]

for state in states:
    populations[state + ", US"] = geocoder.geonames(state, featureCode="ADM1", fuzzy=0.2, key=key).population

for country in countries:
    populations[country] = geocoder.geonames(country, featureClass="A", key=key).population

df = pd.DataFrame(populations.values(), index=populations.keys(), columns=["population"])
df.to_csv("populations.csv")