# FAQ

* How reliable is your data?

I used [this](https://github.com/CSSEGISandData/COVID-19) data from Johns Hopkins University Center for Systems Science and Engineering, who compiled it from data from the World Health Organization, the Chinese CDC, the US CDC and other organizations.

Of course, there could already be a lot more cases going under the radar, in people who are not being tested.

* Why is the Y-axis logarithmic? I want it linear!

A logarithmic Y-axis makes sense for displaying quantities that can be expected to grow exponentially, i.e. with a similar percentage, over time. It allows you to judge relative growth both at the start and later on. This is unlike linear graphs where small variations early on end up at such a small scale as to be completely unnoticeable.

In addition, this allows for better comparison between different series (i.e. countries) that are at different stages. In a linear graph you can't show both a series that is at 100,000 and another at 100 and have both convey meaningful information.

That said, it's easy enough to change this plot to be linear. I included an example in the notebook.

* Where is my favorite country?

I didn't want to graph a hundred countries in one plot, that would be too messy. The easiest way to customize the displayed set of countries is [opening it in Google Colab](https://colab.research.google.com/github/JeroenKools/covid19/blob/master/COVID-19.ipynb), and change the list of countries to include in the interactive plot.

* What criteria did you use to select the countries shown?

Just some of the countries with the highest number of confirmed cases, plus some I have a personal connection to.

* Are you trying to make China look better / my country look worse / scare us?

No. I'm not passing judgment on anyone, or trying to tell anyone to do anything, I'm just looking at data.
