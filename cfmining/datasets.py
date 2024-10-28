import pandas as pd

class Dataset:
    def __init__(self, name):
        self.name = name
        self.outlier_contamination = 0.01
        self.categoric_features = []
        self.not_mutable_features = []

    def __repr__(self):
        return self.name
    
    def load_data(self):
        self.dataframe = pd.read_csv(self.path)
        if not self.use_categorical:
            self.dataframe = self.dataframe.drop(columns = self.categoric_features)
        
        self.mutable_features = self.dataframe.columns
        self.mutable_features = [feat for feat in self.mutable_features if feat not in self.not_mutable_features]
        return self.dataframe


class GermanCredit(Dataset):
    def __init__(self, use_categorical = False):
        super().__init__("german")
        self.use_categorical = use_categorical
        self.outlier_contamination = 0.05
        self.categoric_features = ["PurposeOfLoan"]
        self.target = "GoodCustomer"
        self.not_mutable_features = [
            "Age",
            "OwnsHouse",
            "isMale",
            "JobClassIsSkilled",
            "Single",
            "ForeignWorker",
            "RentsHouse",
        ]
        self.path = "../data/german.csv"    

    
class Taiwan(Dataset):
    def __init__(self, use_categorical = False):
        super().__init__("taiwan")
        self.use_categorical = use_categorical
        self.outlier_contamination = 0.01
        self.categoric_features = ["EDUCATION", "MARRIAGE"]
        self.target = "NoDefaultNextMonth"
        self.not_mutable_features = [
            "Age", "MARRIAGE"
        ]
        self.path = "../data/taiwan.csv"


class Adult(Dataset):
    def __init__(self, use_categorical = False):
        super().__init__("adult")
        self.use_categorical = use_categorical
        self.outlier_contamination = 0.01
        self.categoric_features = ["workclass", "education", "marital_status", "occupation", "relationship", "race", ]
        self.target = "income"
        self.not_mutable_features = ["race", "marital_status", "gender", "age"]
        self.path = "../data/adult.csv"


DATASETS_ = {
    'german': GermanCredit,
    'taiwan': Taiwan,
    'adult': Adult
}