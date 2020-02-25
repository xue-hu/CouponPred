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

# print(os.path.exists(DATA_PATH))
# print(os.path.exists(DATA_PATH/TRAIN_OFF_FILE))
# print(os.path.exists(DATA_PATH/TRAIN_ON_FILE))
# print(os.path.exists(DATA_PATH/SAMPLE_FILE))


class Dataset:
    """
    read in the raw data; convert each col to valid data type; clean the useless samples
    is able to split pos/neg samples and train/test sets
    """
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self._convert_datetime()
        self._clean_nan_value( ['Distance','Discount_rate'], [-1, '1'])
        self.clean_non_numeric_vale(['Discount_rate'])
        self.pos_samples = self._get_positive_sample()
        self.pos_nums = self.pos_samples.shape[0]
        self.neg_samples = self._get_negative_sample()
        self.neg_nums = self.neg_samples.shape[0]
        self.df = pd.concat([self.neg_samples, self.pos_samples])
        self._transform_time_feature()
        # print(self.df.head())
        # print(self.df.info())

    def _int2datetime(self, x):
        if math.isnan(x):
            return None
        y = int(x / 10000)
        m = int(x % 10000 / 100)
        d = int(x % 100)
        return date(y, m, d)

    def _convert_datetime(self):
        if 'Date_received' in self.df.columns:
            self.df['Date_received'] = pd.to_datetime(self.df['Date_received'].apply(self._int2datetime))
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'].apply(self._int2datetime))

    def _convert_discount(self, str):
        if ':' in str:
            discount = str.split(sep=':')
            return (1 - float(discount[1]) / float(discount[0]))
        else:
            return float(str)

    def clean_non_numeric_vale(self, columns):
        for col in columns:
            self.df[col] = self.df[col].apply(self._convert_discount)

    def _clean_nan_value(self, columns, new_vals):
        # print(columns)
        for col,val in zip(columns, new_vals):
            self.df.loc[self.df[col].isnull(), col] = val

    def _get_negative_sample(self):
        neg_df = self.df.loc[~self.df['Coupon_id'].isnull() & self.df['Date'].isnull()]
        pos_df = self.df.loc[~self.df['Coupon_id'].isnull() & ~self.df['Date'].isnull()]
        timeout_df = pos_df.loc[(pos_df['Date'] - pos_df['Date_received']).apply(lambda x: x.days > 15)]
        neg_df = pd.concat([neg_df, timeout_df])
        neg_df['label'] = [0]*neg_df.shape[0]
        return neg_df

    # only 7% are positive samples
    def _get_positive_sample(self):
        pos_df = self.df.loc[~self.df['Coupon_id'].isnull() & ~self.df['Date'].isnull()]
        pos_df = pos_df.loc[(pos_df['Date'] - pos_df['Date_received']).apply(lambda x: x.days <= 15)]
        pos_df['label'] = [1]*pos_df.shape[0]
        return pos_df

    def _transform_time_feature(self):
        self.df['DoW'] = self.df['Date_received'].dt.dayofweek
        self.df['DoM'] = self.df['Date_received'].dt.month
        self.df.drop(['Date', 'Date_received'], axis=1, inplace=True)



if __name__ == '__main__':
    data = Dataset(DATA_PATH/TRAIN_OFF_FILE)
    # print(data.neg_nums)
    # print(data.pos_nums)
