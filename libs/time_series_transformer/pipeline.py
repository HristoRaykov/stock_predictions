from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from libs.time_series_transformer.index_transformer import RowIndexTransformer
from libs.time_series_transformer.lag_creator import LagCreator
from libs.time_series_transformer.log_transformer import LogTransformer
from libs.time_series_transformer.one_hot_encoder import IndicatorVariables
from libs.time_series_transformer.poly_features import PartialPolyFeatures
from libs.time_series_transformer.scaler import DataFrameMinMaxScaler
from libs.time_series_transformer.time_steps import RNNTimeStepsTransformer

LAG_STEPS_NAME_SUFFIX = "_lags"


def create_pipeline(group_col=None, date_col=None, target_col=None,
                    log_trans=False, log_trans_cols=None,
                    create_lags=False, lag_cols=None, lag_num=None, drop_lag_col=False,
                    create_lag_diff=False, lag_diff_num=None,
                    trans_index=False,
                    create_ind_vars=False, not_ind_vars_cols=None,
                    creat_poly_feat=False, poly_cols=None, degree=2, interaction_only=False, include_bias=True,
                    time_step_trans=False,
                    scale=False):
	steps = []
	
	if log_trans:
		step = ("log_transform", LogTransformer(log_trans_cols))
		steps.append(step)
	
	if create_lags:
		for idx, col in enumerate(lag_cols):
			if col == target_col:
				if idx == len(lag_cols) - 1:
					step = (
						col + LAG_STEPS_NAME_SUFFIX,
						LagCreator(lag_col=col, group_col=group_col, lag_num=lag_num,
						           lag_diff=create_lag_diff, lag_diff_num=lag_diff_num, dropna=True))
				else:
					step = (
						col + LAG_STEPS_NAME_SUFFIX,
						LagCreator(lag_col=col, group_col=group_col, lag_num=lag_num,
						           lag_diff=create_lag_diff, lag_diff_num=lag_diff_num))
			else:
				if idx == len(lag_cols) - 1:
					step = (
						col + LAG_STEPS_NAME_SUFFIX,
						LagCreator(lag_col=col, group_col=group_col, lag_num=5,
						           lag_diff=create_lag_diff, lag_diff_num=lag_diff_num,
						           dropna=True, drop_lag_col=drop_lag_col))
				else:
					step = (
						col + LAG_STEPS_NAME_SUFFIX,
						LagCreator(lag_col=col, group_col=group_col, lag_num=5,
						           lag_diff=create_lag_diff, lag_diff_num=lag_diff_num,
						           drop_lag_col=drop_lag_col))
			steps.append(step)
	
	if trans_index:
		idx_trans = RowIndexTransformer(group_col=group_col, date_col=date_col)
		
		step = ("index_trans", idx_trans)
		steps.append(step)
	
	if create_ind_vars:
		step = ("indicator_variables", IndicatorVariables(without_cols=not_ind_vars_cols))
		steps.append(step)
	
	if creat_poly_feat:
		for idx, cols in enumerate(poly_cols):
			if include_bias and idx == 0:
				step = ("poly_features_" + str(idx),
				        PartialPolyFeatures(target_cols=cols, degree=degree, interaction_only=interaction_only,
				                            include_bias=True))
			else:
				step = ("poly_features_" + str(idx),
				        PartialPolyFeatures(target_cols=cols, degree=degree, interaction_only=interaction_only,
				                            include_bias=False))
			steps.append(step)
	
	if time_step_trans:
		time_steps_trans = RNNTimeStepsTransformer(target_col=target_col)
		
		step = ("time_steps_transformer", time_steps_trans)
		steps.append(step)
	
	if scale:
		scaler = DataFrameMinMaxScaler()
		
		step = ("scale", scaler)
		steps.append(step)
	
	return Pipeline(steps)
