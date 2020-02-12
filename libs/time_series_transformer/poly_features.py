import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


class PartialPolyFeatures(PolynomialFeatures):
	
	def __init__(self, target_cols=None, degree=2, interaction_only=False, include_bias=True, order='C'):
		super().__init__(degree, interaction_only, include_bias, order)
		self.target_cols = target_cols
	
	def fit(self, X, y=None):
		return super().fit(X[self.target_cols], y)
	
	def transform(self, X):
		poly = X[self.target_cols]
		
		non_poly_cols = [col for col in X.columns if col not in self.target_cols]
		non_poly = X[non_poly_cols]
		
		poly_feat_names = super().get_feature_names(self.target_cols)
		poly_feat_idxes = poly.index
		poly = super().transform(poly)
		
		poly = pd.DataFrame(poly, columns=poly_feat_names, index=poly_feat_idxes)
		
		temp = pd.concat([non_poly, poly], axis=1)
		
		temp = temp.loc[:, ~temp.columns.duplicated()]
		
		return temp

