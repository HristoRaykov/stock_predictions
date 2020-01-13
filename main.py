import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, TimeDistributed, RepeatVector
from tensorflow.keras.losses import MSE
from tensorflow.keras.metrics import RootMeanSquaredError, MSE
from tensorflow_core.python.keras.backend import concatenate

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.backend import clear_session

clear_session()

DATASET_FILE_PATH = "/home/hristocr/Desktop/Daily Historical Stock Prices (1970 - 2018)/"
HIST_PRICES_FILE_NAME = "historical_stock_prices.csv"
STOCKS_FILE_NAME = "historical_stocks.csv"


# dataset_file = tf.keras.utils.

hist_prices = pd.read_csv(DATASET_FILE_PATH+HIST_PRICES_FILE_NAME)