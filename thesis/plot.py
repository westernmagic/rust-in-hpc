#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib.rcsetup
from matplotlib import rcParams

sns.set()
sns.set_context("paper", 3)
rcParams["figure.figsize"] = (16, 8)
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Open Sans"]
rcParams["axes.prop_cycle"] = matplotlib.rcsetup.cycler("color", [
    "#1F407A",
    "#485A2C",
    "#1269B0",
    "#72791C",
    "#91056A",
    "#6F6F6F",
    "#A8322D",
    "#007A96",
    "#956013",
])

colspecs = [(0, 5), (7, 14), (16, 36), (40, 44), (46, 50), (52, 54), (56, 60), (62, 64), (66, 80), (82, 93)]
names = ["compiler", "language", "version", "nx", "ny", "nz", "num_iter", "num_cpu", "runtime", "error"]

df = pd.read_fwf("data.txt", names = names, header = None, colspecs = colspecs)
print(df.to_latex(index = False, float_format = "%.9f"))
