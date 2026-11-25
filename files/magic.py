import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

columes = [
    "fLength",
    "fWidth",
    "fSize",
    "fConc",
    "fConc1",
    "fAsym",
    "fM3Long",
    "fM3Trans",
    "fAlpha",
    "fDist",
    "class",
]
df = pd.read_csv("../data/magic04.data", names=columes)
df.head(5)

df["class"].unique()
