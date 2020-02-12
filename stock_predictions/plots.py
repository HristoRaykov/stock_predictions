import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from stock_predictions.constants import *


def plot_dist_charts(df, col_names, bins=None, range=None, log=False):
	fig, axes = plt.subplots(1, len(col_names), figsize=(15, 4))
	
	for col_name, ax in list(zip(col_names, axes)):
		values = df[col_name]
		ax.hist(values, bins=bins, log=log, range=range)
		ax.set_xlabel(col_name)
	plt.subplots_adjust(wspace=0.3, hspace=0.1)
	plt.show()


def plot_acf_pacf_by_tickers(data, column=None, tickers=None):
  for ticker in tickers:
    stock_prices = data[data[TICKER_COL]==ticker][column]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    plot_acf(stock_prices, lags=100, ax=ax1)
    ax1.set_xlabel(ticker)

    plot_pacf(stock_prices, ax=ax2)
    ax2.set_xlabel(ticker)

    plt.show()


def plot_model_history(history, fig_title=None, log=False):
	fig, axes = plt.subplots(1, 2, figsize=(12, 6))
	fig.suptitle(fig_title)
	epochs = np.array(history['epoch']).astype(int) + 1
	metrics = np.array(list(history.keys())[1:]).reshape((2, 2)).transpose()
	
	for ax, (train_metric, val_metric) in zip(axes, metrics):
		ax.plot(epochs, history[train_metric], label=train_metric)
		ax.plot(epochs, history[val_metric], label=val_metric)
		if log:
			ax.set_yscale("log")
		ax.legend()
		ax.set_xlabel("epoch")
		ax.set_ylabel(train_metric)

	plt.show()
