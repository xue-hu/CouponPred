import pandas as pd
import os
from datetime import date
import math
from pathlib import Path


DATA_PATH = Path(os.getcwd()).parent
TEST_FILE = r'data/raw/ccf_offline_stage1_test_revised.csv'
TRAIN_OFF_FILE = r'data/raw/ccf_offline_stage1_train.csv'
TRAIN_ON_FILE = r'data/raw/ccf_online_stage1_train.csv'
SAMPLE_FILE = r"data/raw/sample_submission.csv"

print(os.path.exists(DATA_PATH))
print(os.path.exists(DATA_PATH/TRAIN_OFF_FILE))
print(os.path.exists(DATA_PATH/TRAIN_ON_FILE))
print(os.path.exists(DATA_PATH/SAMPLE_FILE))


class Dataset:
    def __init__(self, file_path, ratio):
        df = pd.read_csv(file_path)
        self._convert_datetime(df)
        self._clean_nan_value(df, ['Distance'])
        self.pos_samples = self._get_positive_sample(df)
        self.pos_nums = self.pos_samples.shape[0]
        self.neg_samples = self._get_negative_sample(df)
        self.neg_nums = self.neg_samples.shape[0]
        self.df = pd.concat([self.neg_samples, self.pos_samples])
        self._split_dataset(ratio)

    def _int2datetime(self, x):
        if math.isnan(x):
            return x
        y = int(x / 10000)
        m = int(x % 10000 / 100)
        d = int(x % 100)
        return date(y, m, d)

    def _convert_datetime(self, df):
        if 'Date_received' in df.columns:
            df['Date_received'] = df['Date_received'].apply(self._int2datetime)
        if 'Date' in df.columns:
            df['Date'] = df['Date'].apply(self._int2datetime)

    def _get_negative_sample(self, df):
        neg_df = df.loc[~df['Coupon_id'].isnull() & df['Date'].isnull()]
        pos_df = df.loc[~df['Coupon_id'].isnull() & ~df['Date'].isnull()]
        timeout_df = pos_df.loc[(pos_df['Date'] - pos_df['Date_received']).apply(lambda x: x.days > 15)]
        return pd.concat([neg_df, timeout_df])

    # only 7% are positive samples
    def _get_positive_sample(self, df):
        pos_df = df.loc[~df['Coupon_id'].isnull() & ~df['Date'].isnull()]
        pos_df = pos_df.loc[(pos_df['Date'] - pos_df['Date_received']).apply(lambda x: x.days <= 15)]
        return pos_df

    def _clean_nan_value(self, df, columns):
        # print(columns)
        for col in columns:
            df.loc[df[col].isnull(), col] = -1

    def _split_dataset(self, ratio):
        df = self.df.sample(frac=1).reset_index(drop=True)
        len = self.df.shape[0]
        train_nums = math.ceil(len * ratio)
        self.train_samples = self.df.loc[:train_nums, :]
        self.test_samples = self.df.loc[train_nums:, :]

if __name__ == '__main__':
    ratio = 0.8
    data = Dataset(DATA_PATH/TRAIN_OFF_FILE, ratio)
    print(data.neg_nums)
    print(data.pos_nums)
    print(data.train_samples.shape[0])
    print(data.test_samples.shape[0])
