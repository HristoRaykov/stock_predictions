import pandas as pd

from sklearn.preprocessing import MinMaxScaler


class DataFrameMinMaxScaler(MinMaxScaler):
	
	def __init__(self, feature_range=(0, 1), copy=True):
		super().__init__(feature_range, copy)
	
	def fit(self, X, y=None):
		return super().fit(X, y)
	
	def transform(self, X):
		col_names = X.columns
		indexes = X.index
		
		temp = super().transform(X)
		temp = pd.DataFrame(temp, columns=col_names, index=indexes)
		
		return temp
	
	def inverse_transform(self, X):
		col_names = X.columns
		indexes = X.index
	
		temp = super().inverse_transform(X)
		temp = pd.DataFrame(temp, columns=col_names, index=indexes)
	
		return temp

# class DataFrameMinMaxScaler(MinMaxScaler):
#
# 	def __init__(self, without_cols=None, feature_range=(0, 1), copy=True):
# 		super().__init__(feature_range, copy)
# 		self.without_cols = without_cols
#
# 	def fit(self, X, y=None):
# 		if self.without_cols is None:
# 			temp = X
# 		else:
# 			temp = X.drop(self.without_cols, axis=1)
#
# 		return super().fit(temp, y)
#
# 	def transform(self, X):
# 		if self.without_cols is None:
# 			not_scaled = pd.DataFrame()
# 			temp = X
# 		else:
# 			not_scaled = pd.DataFrame(X[self.without_cols])
# 			temp = X.drop(self.without_cols, axis=1)
#
# 		col_names = temp.columns
# 		indexes = temp.index
#
# 		temp = super().transform(temp)
#
# 		temp = pd.DataFrame(temp, columns=col_names, index=indexes)
#
# 		temp = pd.concat([temp, not_scaled], axis=1)
#
# 		return temp
