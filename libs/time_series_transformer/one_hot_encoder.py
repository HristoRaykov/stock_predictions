import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class IndicatorVariables(BaseEstimator, TransformerMixin):
	
	def __init__(self, without_cols=None):
		self.without_cols = without_cols
	
	def fit(self, X, y=None):
		return self
	
	def transform(self, X):
		
		if self.without_cols is None:
			not_transform = pd.DataFrame()
			temp = X
		else:
			not_transform = pd.DataFrame(X[self.without_cols])
			temp = X.drop(self.without_cols, axis=1)
		
		# col_names = temp.columns
		# indexes = temp.index
		
		temp = pd.get_dummies(temp, drop_first=True)
		
		# temp = pd.DataFrame(temp, columns=col_names, index=indexes)
		
		temp = pd.concat([temp, not_transform], axis=1)
		
		return temp
