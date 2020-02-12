import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class LogTransformer(BaseEstimator, TransformerMixin):
	
	def __init__(self, cols):
		self.cols = cols
	
	def fit(self, X, y=None):
		return self
	
	def transform(self, X):
		temp = X.copy()
		for col in self.cols:
			mask = (temp[col] >= 0).replace({0: -1, 1: 1}).astype(int)
			pos_vals = temp[col] * mask
			pos_vals = np.log1p(pos_vals)
			log_vals = pos_vals * mask
			temp[col] = log_vals
		
		return temp
