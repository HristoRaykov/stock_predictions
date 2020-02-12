import pandas as pd
import ta

BB_MA_COL = "bb_ma"
BB_HIGH_BAND_COL = "bb_hb"
BB_HIGH_BAND_IND_COL = "bb_hb_ind"
BB_LOW_BAND_COL = "bb_lb"
BB_LOW_BAND_IND_COL = "bb_lb_ind"

ATR_COL = "atr"

STD_COL = "std"


def calculate_volatility_indicators(df, low_col=None, high_col=None, close_col=None):
	df = calculate_bb(df, close_col=close_col)
	df = calculate_atr(df, low_col=low_col, high_col=high_col, close_col=close_col)
	df = calculate_std(df, close_col=close_col)
	
	return df


def calculate_bb(df, close_col=None):
	bb = ta.volatility.BollingerBands(close=df[close_col])
	df[BB_MA_COL] = bb.bollinger_mavg()
	df[BB_HIGH_BAND_COL] = bb.bollinger_hband()
	df[BB_HIGH_BAND_IND_COL] = bb.bollinger_hband_indicator()
	df[BB_LOW_BAND_COL] = bb.bollinger_lband()
	df[BB_LOW_BAND_IND_COL] = bb.bollinger_lband_indicator()
	
	return df


def calculate_atr(df, low_col=None, high_col=None, close_col=None):
	atr = ta.volatility.AverageTrueRange(high=df[high_col], low=df[low_col], close=df[close_col])
	df[ATR_COL] = atr.average_true_range()
	
	return df


def calculate_std(df, close_col=None, window=14):
	df[STD_COL] = df[close_col].rolling(window).std()
	
	return df
