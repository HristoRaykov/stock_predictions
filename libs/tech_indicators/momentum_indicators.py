import pandas as pd
import ta

STOCH_OSC_COL = "stoch_osc"
STOCH_OSC_SIGNAL_COL = "stoch_osc_signal"

RSI_COL = "rsi"

ROC_COL = "roc"


def calculate_momentum_indicators(df, low_col=None, high_col=None, close_col=None):
	df = calculate_stoch_oscillator(df, low_col=low_col, high_col=high_col, close_col=close_col)
	df = calculate_rsi(df, close_col=close_col)
	df = calculate_rate_of_change(df, close_col=close_col)
	
	return df


def calculate_stoch_oscillator(df, low_col=None, high_col=None, close_col=None):
	stoch_osc = ta.momentum.StochasticOscillator(high=df[high_col], low=df[low_col], close=df[close_col])
	df[STOCH_OSC_COL] = stoch_osc.stoch()
	df[STOCH_OSC_SIGNAL_COL] = stoch_osc.stoch_signal()
	
	return df


def calculate_rsi(df, close_col=None):
	rsi = ta.momentum.RSIIndicator(close=df[close_col])
	df[RSI_COL] = rsi.rsi()
	
	return df


def calculate_rate_of_change(df, close_col=None, window=1):
	df[ROC_COL] = (df[close_col].diff(window) / df[close_col].shift(window)) * 100
	
	return df
