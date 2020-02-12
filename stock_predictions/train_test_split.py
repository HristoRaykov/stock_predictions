def split_time_series_by_time_steps_column(df, time_step_col=None, n_time_steps=None):
	time_steps = df[time_step_col].unique()
	split_time_step = time_steps[-n_time_steps]
	train = df[df["date"] < split_time_step]
	test = df[df["date"] >= split_time_step]
	
	return train, test


def split_time_series_by_time_steps_index(df, n_time_steps=None):
	"""
	df index is Multiindex with level_0='stock tickers' and level_1='dates'
	"""
	
	time_steps = df.index.get_level_values(1).unique()
	split_time_step = time_steps[-n_time_steps]
	train = df[df.index.get_level_values(1) < split_time_step]
	test = df[df.index.get_level_values(1) >= split_time_step]
	
	return train, test
