import numpy as np
import pandas as pd

def preprocess_german():
    df = pd.read_csv("../data/german_raw.csv")
    df["is_male"] = df.Gender == "Male"
    df["is_male"] = df.is_male.astype(int)

    df = df.drop(columns = ["Gender"])
    # remove OtherLoansAtStore because only has one value
    df = df.drop(columns = ["OtherLoansAtStore"])
    df["PurposeOfLoan"] = df["PurposeOfLoan"].astype("category")
    df["GoodCustomer"] = df["GoodCustomer"].map({1: 1, -1: 0})
    df.to_csv("../data/german.csv", index = False)

def preprocess_taiwan():
    # taiwan
    df = pd.read_csv("../data/credit_raw.csv")
    processed_df = pd.DataFrame()

    # convert NTD to USD using spot rate in 09-2005
    NTD_to_USD = 32.75 # see https://www.poundsterlinglive.com/bank-of-england-spot/historical-spot-exchange-rates/usd/USD-to-TWD-2005
    monetary_features = list(filter(lambda x: ('BILL_AMT' in x) or ('PAY_AMT' in x) or ('LIMIT_BAL' in x), df.columns))
    processed_df[monetary_features] = df[monetary_features].applymap(lambda x: x / NTD_to_USD).round(-1).astype(int)


    # outcome variable in first column
    processed_df['NoDefaultNextMonth'] = 1.0 - df['default payment next month']

    # Age
    processed_df['Age'] = df['AGE']


    # Process Bill Related Variables
    pay_columns = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    bill_columns = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']



    #processed_df['LastBillAmount'] = np.maximum(df['BILL_AMT1'], 0)
    processed_df['MaxBillAmountOverLast6Months'] = np.maximum(df[bill_columns].max(axis = 1), 0)
    processed_df['MaxPaymentAmountOverLast6Months'] = np.maximum(df[pay_columns].max(axis = 1), 0)
    processed_df['MonthsWithZeroBalanceOverLast6Months'] = np.sum(np.greater(df[pay_columns].values, df[bill_columns].values), axis = 1)
    processed_df['MonthsWithLowSpendingOverLast6Months'] = np.sum(df[bill_columns].div(df['LIMIT_BAL'], axis = 0) < 0.20, axis = 1)
    processed_df['MonthsWithHighSpendingOverLast6Months'] = np.sum(df[bill_columns].div(df['LIMIT_BAL'], axis = 0) > 0.80, axis = 1)
    processed_df['MostRecentBillAmount'] = np.maximum(df[bill_columns[0]], 0)
    processed_df['MostRecentPaymentAmount'] = np.maximum(df[pay_columns[0]], 0)



    # Credit History
    # PAY_M' = months since last payment (as recorded last month)
    # PAY_6 =  months since last payment (as recorded 6 months ago)
    # PAY_M = -1 if paid duly in month M
    # PAY_M = -2 if customer was issued refund M
    df = df.rename(columns = {'PAY_0': 'MonthsOverdue_1',
                                    'PAY_2': 'MonthsOverdue_2',
                                    'PAY_3': 'MonthsOverdue_3',
                                    'PAY_4': 'MonthsOverdue_4',
                                    'PAY_5': 'MonthsOverdue_5',
                                    'PAY_6': 'MonthsOverdue_6'})

    overdue = ['MonthsOverdue_%d' % j for j in range(1, 7)]
    df[overdue] = df[overdue].replace(to_replace = [-2, -1], value = [0, 0])
    overdue_history = df[overdue].values > 0
    payment_history = np.logical_not(overdue_history)

    def count_zero_streaks(a):
        #adapted from zero_runs function of https://stackoverflow.com/a/24892274/568249
        iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        runs = np.where(absdiff == 1)[0].reshape(-1, 2)
        n_streaks = runs.shape[0]
        #streak_lengths = np.sum(runs[:,1] - runs[:,0])
        return n_streaks


    overdue_counts = np.repeat(np.nan, len(df))
    n_overdue_months = np.sum(overdue_history > 0, axis = 1)
    overdue_counts[n_overdue_months == 0] = 0 # count_zero_streaks doesn't work for edge cases
    overdue_counts[n_overdue_months == 6] = 1


    for k in range(1, len(overdue)):
        idx = n_overdue_months == k
        overdue_counts[idx] = [count_zero_streaks(a) for a in payment_history[idx, :]]

    overdue_counts = overdue_counts.astype(np.int_)
    processed_df['TotalOverdueCounts'] = overdue_counts
    processed_df['TotalMonthsOverdue'] = df[overdue].sum(axis = 1)
    processed_df['HistoryOfOverduePayments'] = df[overdue].sum(axis = 1) > 0


    cat_features = ["EDUCATION", "MARRIAGE"]
    for feature in cat_features:
        processed_df[feature] = df[feature].astype("category")

    processed_df.to_csv("../data/taiwan.csv", index = False)


def preprocess_adult():
    df = pd.read_csv("../data/adult_raw.csv")
    df = df.drop(columns = ["fnlwgt", "educational-num", "native-country"])
    df.columns = [col.replace("-", "_") for col in df.columns]
    df["income"] = df["income"].map({">50K": 1, "<=50K": 0})
    df.to_csv("../data/adult.csv", index = False)
    

if __name__ == "__main__":
    preprocess_german()
    preprocess_taiwan()
    preprocess_adult()
