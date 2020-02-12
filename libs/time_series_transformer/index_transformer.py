import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class RowIndexTransformer(BaseEstimator, TransformerMixin):
	
	def __init__(self, group_col=None, date_col=None):
		self.group_col = group_col
		self.date_col = date_col
	
	def fit(self, X, y=None):
		return self
	
	def transform(self, X):
		index = pd.MultiIndex.from_frame(X[[self.group_col, self.date_col]])
		temp = X.drop([self.group_col, self.date_col], axis=1)
		temp = temp.set_index(index)
		
		return temp
