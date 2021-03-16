#!/usr/bin/env python

import pandas as pd
import csv

colspecs = [(0, 5), (7, 14), (16, 36), (40, 44), (46, 50), (52, 54), (56, 60), (62, 64), (66, 80), (82, 93)]
names = ["compiler", "language", "version", "nx", "ny", "nz", "num_iter", "num_cpu", "runtime", "error"]

df = pd.read_fwf("data.txt", names = names, header = None, colspecs = colspecs)
print(
    df
    .sort_values(by=["nx", "ny", "num_cpu", "language", "compiler", "version"])
    .drop(["ny", "nz", "num_iter"], axis=1)
    .to_csv(index=False, float_format="%.9f", header=False, sep="&", line_terminator="\\\\\n", quoting=csv.QUOTE_NONE)
    .replace("_", "\\_")
)
