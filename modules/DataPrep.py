from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import pandas as pd


def prepareDataset(dataset):
    # read in dataset
    df = pd.read_csv(dataset)
    # drop features that has no significant value
    df = df.drop(['SourceIP', 'PacketTimeMode',
                 'DestinationIP', 'TimeStamp'], 1)

    # dropping empty rows
    df = df.dropna()

    # drop duplicate information
    df = df.drop_duplicates()

    # encoding target variable
    df.DoH = LabelEncoder().fit_transform(df.DoH)

    # balance majority and minority class
    df_majority = df[df.DoH == 0]
    df_minority = df[df.DoH == 1]

    # downsampling the majority class
    df_majority_downsampled = resample(df_majority, replace=False,
                                       n_samples=16199, random_state=42)

    df = pd.concat([df_majority_downsampled, df_minority])

    # separate X and y variables
    X = df.drop('DoH', 1)
    y = df.DoH

    return X, y
