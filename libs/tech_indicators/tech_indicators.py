# Types:
#   - Trend Indicators
#       -- moving averages - lagging
#       -- MACD - - lagging
#       -- Parabolic Stop And Reverse (Parabolic SAR) - leading
#   - Momentum Indicators
#       --  Stochastick Oscillator - leading
#       --  Commodity Channel Index (CCI) - leading
#       --  Relative Strength Index (RSI) - leading
#   - Volatility Indicators
#       --  Bollinger Bands - lagging
#       --  Average True Range - lagging
#       --  Standard Deviation - lagging
#   - Volume Indicators
#       --  Chaikin Oscillator - leading
#       --  On-Balance Volume (OBV) - leading
#       --  Volume Rate Of Change - lagging
#   - Support and Resistance
#       -- Fibonacci
#
#
# Class:
#   - Overlays
#   - Oscillators
#   - Cumulatives
#   - Index
#
#
#
#
#
#   - Leading - trend is about to start (Stochastic, RSI)
#   - Lagging - follow the price action
#   - Confirming
#

from libs.tech_indicators.trend_indicators import calculate_trend_indicators
from libs.tech_indicators.momentum_indicators import calculate_momentum_indicators
from libs.tech_indicators.volatility_indicators import calculate_volatility_indicators
from libs.tech_indicators.volume_indicators import calculate_volume_indicators


def calculate_tech_indicators(df, low_col=None, high_col=None, close_col=None, volume_col=None):
	df = calculate_trend_indicators(df, low_col=low_col, high_col=high_col, close_col=close_col)
	df = calculate_momentum_indicators(df, low_col=low_col, high_col=high_col, close_col=close_col)
	df = calculate_volatility_indicators(df, low_col=low_col, high_col=high_col, close_col=close_col)
	df = calculate_volume_indicators(df, low_col=low_col, high_col=high_col, close_col=close_col, volume_col=volume_col)
	
	return df
