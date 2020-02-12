import pandas as pd
import ta

EMA_COL = "ema"

MACD_COL = "macd"
MACD_DIFF_COL = "macd_diff"
MACD_SIGNAL_COL = "macd_signal"

PSAR_COL = "psar"
PSAR_DOWN_COL = "psar_down"
PSAR_DOWN_IND_COL = "psar_down_ind"
PSAR_UP_COL = "psar_up"
PSAR_UP_IND_COL = "psar_up_ind"

CCI_COL = "cci"


def calculate_trend_indicators(df, low_col=None, high_col=None, close_col=None):
	df = calculate_ema(df, close_col=close_col)
	df = calculate_macd(df, close_col=close_col)
	df = calculate_psar(df, low_col=low_col, high_col=high_col, close_col=close_col)
	df = calculate_cci(df, low_col=low_col, high_col=high_col, close_col=close_col)
	
	return df


def calculate_ema(df, close_col=None):
	ema = ta.trend.EMAIndicator(close=df[close_col])
	df[EMA_COL] = ema.ema_indicator()
	
	return df


def calculate_macd(df, close_col=None):
	macd = ta.trend.MACD(close=df[close_col])
	df[MACD_COL] = macd.macd()
	df[MACD_DIFF_COL] = macd.macd_diff()
	df[MACD_SIGNAL_COL] = macd.macd_signal()
	
	return df


def calculate_psar(df, low_col=None, high_col=None, close_col=None):
	psar = ta.trend.PSARIndicator(high=df[high_col], low=df[low_col], close=df[close_col])
	df[PSAR_COL] = psar.psar()
	# df[PSAR_UP_COL] = psar.psar_up()
	df[PSAR_UP_IND_COL] = psar.psar_up_indicator()
	# df[PSAR_DOWN_COL] = psar.psar_down()
	df[PSAR_DOWN_IND_COL] = psar.psar_down_indicator()
	
	return df


def calculate_cci(df, low_col=None, high_col=None, close_col=None):
	cci = ta.trend.CCIIndicator(high=df[high_col], low=df[low_col], close=df[close_col])
	df[CCI_COL] = cci.cci()
	
	return df
