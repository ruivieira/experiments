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

def get_result_schema():
    return ['ExternalRiskEstimate', 'MSinceOldestTradeOpen',
       'MSinceMostRecentTradeOpen', 'AverageMInFile', 'NumSatisfactoryTrades',
       'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec',
       'PercentTradesNeverDelq', 'MSinceMostRecentDelq',
       'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumTotalTrades',
       'NumTradesOpeninLast12M', 'PercentInstallTrades',
       'MSinceMostRecentInqexcl7days', 'NumInqLast6M', 'NumInqLast6Mexcl7days',
       'NetFractionRevolvingBurden', 'NetFractionInstallBurden',
       'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance',
       'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance',
       'CfExternalRiskEstimate', 'CfMSinceOldestTradeOpen',
       'CfMSinceMostRecentTradeOpen', 'CfAverageMInFile',
       'CfNumSatisfactoryTrades', 'CfNumTrades60Ever2DerogPubRec',
       'CfNumTrades90Ever2DerogPubRec', 'CfPercentTradesNeverDelq',
       'CfMSinceMostRecentDelq', 'CfMaxDelq2PublicRecLast12M', 'CfMaxDelqEver',
       'CfNumTotalTrades', 'CfNumTradesOpeninLast12M',
       'CfPercentInstallTrades', 'CfMSinceMostRecentInqexcl7days',
       'CfNumInqLast6M', 'CfNumInqLast6Mexcl7days',
       'CfNetFractionRevolvingBurden', 'CfNetFractionInstallBurden',
       'CfNumRevolvingTradesWBalance', 'CfNumInstallTradesWBalance',
       'CfNumBank2NatlTradesWHighUtilization', 'CfPercentTradesWBalance',
       'GoalValue', 'GoalName', 'GoalScore', 'method']

def save_result(original, cf, score, method: str, model: str):
    result = {}
    result.update(original)
    result.update(cf)
    result.update(score)
    result.update({"method": method})
    result.update({"model": model})
    result_df = pd.DataFrame.from_dict(result)
    return result_df