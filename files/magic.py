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
df["class"] = (df["class"] == "g").astype(int)

for label in columes[:-1]:
    plt.hist(
        df[df["class"] == 1][label],
        color="blue",
        label="gamma",
        alpha=0.8,
        density=True,
    )
    plt.hist(
        df[df["class"] == 0][label],
        color="red",
        label="hadron",
        alpha=0.8,
        density=True,
    )
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()
