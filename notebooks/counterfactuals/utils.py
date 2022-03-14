import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "RiskPerformance"


def data_bounds(df):
    df = pd.read_csv("../../datasets/FICO/heloc_dataset_v1.csv")
    summary = df.describe().T
    bounds = summary[["min", "max"]].to_dict()
    return bounds


def get_negative_closest(model, probability):
    df = pd.read_csv("../../datasets/FICO/heloc_dataset_v1.csv")
    df[TARGET] = df[TARGET].factorize()[0]

    train, test = train_test_split(df, test_size=0.25, random_state=42)

    train_x = train[df.columns[~train.columns.isin([TARGET])]]
    train_y = train[TARGET]

    test_x = test[test.columns[~test.columns.isin([TARGET])]]
    test_y = test[TARGET]

    probabilities = model.predict_proba(test_x)

    import numpy as np

    bad_probs = np.abs(probabilities[:, 0] - probability)
    index = bad_probs.argmin()

    return test_x.iloc[index]