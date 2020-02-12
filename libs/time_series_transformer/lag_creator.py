from sklearn.base import BaseEstimator, TransformerMixin

from libs.time_series_transformer.constants import LAG_SUFFIX_KW, LAG_DIFF_SUFFIX_KW


class LagCreator(BaseEstimator, TransformerMixin):
	
	def __init__(self, lag_col=None, group_col=None, lag_num=None, lag_diff=False, lag_diff_num=1, dropna=False,
	             drop_lag_col=False):
		self.target_col = lag_col
		self.group_col = group_col
		self.lag_num = lag_num
		self.dropna = dropna
		self.drop_lag_col = drop_lag_col
		self.lag_diff = lag_diff
		self.lag_diff_num = lag_diff_num
	
	def fit(self, X, y=None):
		return self
	
	def transform(self, X):
		tmp = X.copy()
		for lags_count in range(1, self.lag_num + 1):
			lag_col_name = self.target_col + LAG_SUFFIX_KW + str(lags_count)
			tmp[lag_col_name] = tmp.groupby([self.group_col])[self.target_col].shift(lags_count)
			
			if self.lag_diff:
				for lags_diff_count in range(1, self.lag_diff_num + 1):
					lag_diff_col_name = lag_col_name + LAG_DIFF_SUFFIX_KW + str(lags_diff_count)
					tmp[lag_diff_col_name] = tmp.groupby([self.group_col])[lag_col_name].diff(lags_diff_count)
		
		if self.dropna:
			tmp = tmp.dropna()
		
		if self.drop_lag_col:
			tmp.drop([self.target_col], axis=1, inplace=True)
		
		return tmp
