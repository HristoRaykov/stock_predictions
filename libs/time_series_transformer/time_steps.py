import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from libs.time_series_transformer.constants import LAG_SUFFIX_KW


class RNNTimeStepsTransformer(BaseEstimator, TransformerMixin):
	
	def __init__(self, target_col=None):
		self.target_col = target_col
	
	def fit(self, X, y=None):
		return self
	
	def transform(self, X):
		dataset = pd.DataFrame(X[self.target_col].values,
		                       columns=pd.MultiIndex.from_arrays([[LAG_SUFFIX_KW[1:] + "0"], [self.target_col]]),
		                       index=X.index)
		cols = list(X.columns)
		
		lag_cols = self._extract_lag_cols(cols)
		time_ind_cols = self._extract_time_ind_cols(cols)
		
		dataset = self._transform(X, dataset, lag_cols, time_ind_cols)
		
		return dataset
	
	def _extract_time_ind_cols(self, cols):
		time_ind_cols = [col for col in cols if
		                 LAG_SUFFIX_KW not in col and col != self.target_col]
		
		return time_ind_cols
	
	@staticmethod
	def _extract_lag_cols(cols):
		return [col for col in cols if LAG_SUFFIX_KW in col]
	
	@staticmethod
	def _extract_lag_nums(lag_cols):
		numbers = {d for col in lag_cols for d in col if d.isdigit()}
		numbers = sorted(numbers)
		
		return numbers
	
	def _transform(self, X, dataset, lag_cols, time_ind_cols):
		lag_nums = self._extract_lag_nums(lag_cols)
		index = dataset.index
		
		for i in lag_nums:
			time_step = LAG_SUFFIX_KW[1:] + i
			lag_i_cols = [col for col in lag_cols if time_step in col]
			lag_i_cols.extend(time_ind_cols)
			lag_i_df = X[lag_i_cols]
			lag_i_df = lag_i_df.set_index(index)
			lag_i_df.columns = pd.MultiIndex.from_product([[time_step], lag_i_cols])
			
			dataset = pd.concat([dataset, lag_i_df], axis=1)
		
		return dataset
