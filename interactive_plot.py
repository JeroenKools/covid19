from covid19_processing import *
import ipywidgets as widgets


def run():
    data = Covid19Processing(False)
    data.process(rows=0, debug=False)
    data.set_default_countries([
        "World", "California, US", "Mongolia", "United States",
        "India", "Netherlands"])
    widgets.interact(data.plot_interactive,
                 x_metric=["calendar_date", "day_number"],
                 y_metric=["confirmed", "deaths", "active",
                           "new confirmed", "new deaths",
                           "confirmed/population", "active/population", "deaths/population",
                           "new confirmed/population", "new deaths/population",
                           "recent confirmed", "recent deaths",
                           "recent confirmed/population", "recent deaths/population",
                           "deaths/confirmed", "recent deaths/recent confirmed"],
                 smoothing_days=widgets.IntSlider(min=0, max=31, step=1, value=7), use_log_scale=True)
