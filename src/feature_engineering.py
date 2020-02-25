import tensorflow as tf
from pre_process import Dataset
from pathlib import Path
import os

class FeatureEng:
    def  __init__(self, df):
        labels = df.pop('label')
        dataset = tf.data.Dataset.from_tensor_slices((df.values, labels.values))
        for feat, label in dataset.take(2):
            print(feat)
            print(label)


if __name__ == '__main__':
    DATA_PATH = Path(os.getcwd()).parent
    TRAIN_OFF_FILE = r'data/raw/ccf_offline_stage1_train.csv'
    data = Dataset(DATA_PATH / TRAIN_OFF_FILE)
    eng = FeatureEng(data.df)