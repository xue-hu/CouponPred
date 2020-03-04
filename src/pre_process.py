import pandas as pd
import os
from datetime import date
import math
from pathlib import Path
import tensorflow as tf

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
    def __init__(self, file_path, ratio=0.9):
        self.df = pd.read_csv(file_path)
        self._convert_datetime()
        self._clean_nan_value( ['Distance','Discount_rate'], [-1, '1'])
        self._clean_non_numeric_value(['Discount_rate'])
        self.pos_samples = self._get_positive_sample()
        self.pos_nums = self.pos_samples.shape[0]
        self.neg_samples = self._get_negative_sample()
        self.neg_nums = self.neg_samples.shape[0]
        self.df = pd.concat([self.neg_samples, self.pos_samples])
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.df = self.df.astype({"Coupon_id": int})
        self._transform_time_feature()
        len = int(self.df.shape[0] * ratio)
        self.train_df = self.df[:len]
        self.test_df = self.df[len:]
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

    def _clean_non_numeric_value(self, columns):
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

    def generate_feature_cols(self):
        cat_cols = ['User_id', 'Merchant_id', 'Coupon_id', 'DoW', 'DoM']
        num_cols = ['Discount_rate', 'Distance']
        one_hot_cols = [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_hash_bucket
                                                           ('Merchant_id', hash_bucket_size=8900,
                                                            dtype=tf.dtypes.int64)) ]
        one_hot_cols += [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_hash_bucket
                                                            ('Coupon_id', hash_bucket_size=15000,
                                                             dtype=tf.dtypes.int64))]
        one_hot_cols += [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_hash_bucket
                                                            ('User_id', hash_bucket_size=7400000,
                                                             dtype=tf.dtypes.int64))]
        # one_hot_cols += [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_hash_bucket
        #                                                     (col, hash_bucket_size=7400000,
        #                                                      dtype=tf.dtypes.int64)) for col in ['DoW', 'DoM']]
        feature_cols = [tf.feature_column.numeric_column(k) for k in num_cols]
        return feature_cols + one_hot_cols


if __name__ == '__main__':
    data = Dataset(DATA_PATH/TRAIN_OFF_FILE)
    # print(data.neg_nums)
    # print(data.pos_nums)
