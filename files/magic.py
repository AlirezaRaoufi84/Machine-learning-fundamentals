import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

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

# train, validation, testing data


train, temp = train_test_split(df, test_size=0.4, random_state=42)
valid, test = train_test_split(temp, test_size=0.5, random_state=42)


def scale_dataset(df, over_sample=False):
    x = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values

    if over_sample:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x, y)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    data = np.hstack((x, np.reshape(y, (-1, 1))))
    return x, y, data


# before scale
print(len(train[train["class"] == 1]))  # gammas 7396
print(len(train[train["class"] == 0]))  # hadron 4016

x_train, y_train, train = scale_dataset(train, over_sample=True)
x_valid, y_valid, valid = scale_dataset(valid, over_sample=False)
x_test, y_test, test = scale_dataset(test, over_sample=False)

# after scale
sum(y_train == 1)  # 7396
sum(y_train == 0)  # 7396

# Data is now completely formatted and we are now ready to train the model.
