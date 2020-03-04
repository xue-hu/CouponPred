import tensorflow as tf
from pre_process import Dataset
from pathlib import Path
import os
import logging

logging.basicConfig(level=logging.DEBUG)

class Executor:
    def  __init__(self,  data, batch_size):
        self.batch_size =batch_size
        self.data = data
        self.construct_estimator()

    def input_fn(self, mode='train'):
        if mode == 'train':
            df = self.data.train_df.copy()
            count = None
        else:
            df = self.data.test_df.copy()
            count = 1
        labels = df.pop('label')
        dataset = tf.data.Dataset.from_tensor_slices((df.to_dict('list'), labels.values)).shuffle(100)
        dataset = dataset.repeat(count = count).batch(self.batch_size)
        return dataset


    def model_fn(self):
        pass

    def construct_estimator(self):
        feature_columns = self.data.generate_feature_cols()
        self.estimator = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[100, 100], model_dir='./model')

    def run(self):
        logging.debug('training...')
        self.estimator.train(input_fn=lambda: self.input_fn(), steps=10000)
        # Evaluate loss over one epoch of test_set.
        logging.debug('evaluating...')
        ev = self.estimator.evaluate(input_fn=lambda: self.input_fn( mode='eval'))
        loss_score = ev["loss"]
        print("Loss: {0:f}".format(loss_score))

    def train_eval(self):
        logging.debug('train and eval...')
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: self.input_fn(), max_steps=100000)
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: self.input_fn(mode='eval'))
        tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)


if __name__ == '__main__':
    DATA_PATH = Path(os.getcwd()).parent
    TRAIN_OFF_FILE = r'data/raw/ccf_offline_stage1_train.csv'
    data = Dataset(DATA_PATH / TRAIN_OFF_FILE)
    exe = Executor(data, 2)
    # exe.run(data)
    exe.train_eval()