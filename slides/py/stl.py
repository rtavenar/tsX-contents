# Based on statsmodels docs

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.seasonal import STL

register_matplotlib_converters()
sns.set_style("darkgrid")

plt.rc("figure", figsize=(20, 12))
plt.rc("font", size=20)

co2 = pd.read_csv(
    "https://raw.githubusercontent.com/statsmodels/smdatasets/refs/heads/main/data/stl-decomposition/co2.csv",
    parse_dates=True,
    index_col=0,
).iloc[:, 0]


stl = STL(co2, seasonal=13)
res = stl.fit()
fig = res.plot()
plt.savefig("slides/fig/stl.svg")
