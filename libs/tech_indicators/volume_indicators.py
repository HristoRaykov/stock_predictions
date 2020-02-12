import numpy as np
import pandas as pd
import ta

CHAIKIN_MONEY_FLOW_COL = "cmf"
ON_BALANCE_VOL_COL = "obv"
VWAP = "vwap"


def calculate_volume_indicators(df, low_col=None, high_col=None, close_col=None, volume_col=None):
	df = calculate_chaikin_money_flow(df, low_col=low_col, high_col=high_col, close_col=close_col,
	                                  volume_col=volume_col)
	df = calculate_obv(df, close_col=close_col, volume_col=volume_col)
	df = calculate_vwap(df, close_col=close_col, volume_col=volume_col)
	
	return df


def calculate_chaikin_money_flow(df, low_col=None, high_col=None, close_col=None, volume_col=None):
	cmf = ta.volume.ChaikinMoneyFlowIndicator(high=df[high_col], low=df[low_col], close=df[close_col],
	                                          volume=df[volume_col])
	df[CHAIKIN_MONEY_FLOW_COL] = cmf.chaikin_money_flow()
	
	return df


def calculate_obv(df, close_col=None, volume_col=None):
	obv = ta.volume.OnBalanceVolumeIndicator(close=df[close_col], volume=df[volume_col])
	df[ON_BALANCE_VOL_COL] = obv.on_balance_volume()
	
	return df


def calculate_vwap(df, close_col=None, volume_col=None):
	df[VWAP] = (df[close_col] * df[volume_col]).cumsum() / df[volume_col].cumsum()
	
	return df
