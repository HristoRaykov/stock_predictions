import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow.keras.metrics import RootMeanSquaredError, MSE, Accuracy


def inv_scale_rmse(y_true, y_pred, scaler=None):
	y_true_rescaled = scaler.inverse_transform(y_true)
	y_pred_rescaled = scaler.inverse_transform(y_pred)

	return np.sqrt(MSE(y_true_rescaled.reshape(-1, ), y_pred_rescaled.reshape(-1, )))


def price_direction_accuracy(targets_directs, predicts_directs):
	accur = Accuracy()
	accur.update_state(targets_directs, predicts_directs)
	
	return accur.result().numpy()


def calculate_price_directions(targets, predictions):
  price_directs = targets.copy()
  price_directs["targets_lag_1"] = price_directs.groupby(level=0).shift()
  price_directs["targets_price_changes"] = price_directs[("lag_0", "adj_close")] - price_directs["targets_lag_1"]

  price_directs["predictions"] = predictions
  price_directs["predictions_price_changes"] = price_directs["predictions"] - price_directs["targets_lag_1"]

  price_directs = price_directs.dropna()
  price_directs["targets_directions"] = (price_directs["targets_price_changes"] >= 0).astype(int)
  price_directs["predictions_directions"] = (price_directs["predictions_price_changes"] >= 0).astype(int)

  return price_directs["targets_directions"].values, price_directs["predictions_directions"].values
  

def calculate_model_metrics(model, history, train_data=None, train_targets=None, val_data=None, val_targets=None, test_data=None, test_targets=None, scaler=None):
  metrics = pd.DataFrame(index=[model.name])
  metrics["train_epochs"] = [history["epoch"][-1] + 1]
  metrics["train_mse"] = [history["loss"][-1]]
  metrics["train_rmse"] = [history["root_mean_squared_error"][-1]]

  train_pred = model.predict(train_data)
  metrics["train_inv_scale_rmse"] = [inv_scale_rmse(train_targets, train_pred, scaler=scaler)]

  targets_train_directs, predicts_train_directs = calculate_price_directions(train_targets, train_pred)
  metrics["train_price_direct_accuracy"] = [price_direction_accuracy(targets_train_directs, predicts_train_directs)]


  metrics["val_mse"] = [history["val_loss"][-1]]
  metrics["val_rmse"] = [history["val_root_mean_squared_error"][-1]]

  val_pred = model.predict(val_data)
  metrics["val_inv_scale_rmse"] = [inv_scale_rmse(val_targets, val_pred, scaler=scaler)]

  targets_val_directs, predicts_val_directs = calculate_price_directions(val_targets, val_pred)
  metrics["val_price_direct_accuracy"] = [price_direction_accuracy(targets_val_directs, predicts_val_directs)]


  test_pred = model.predict(test_data)
  metrics["test_inv_scale_rmse"] = [inv_scale_rmse(test_targets, test_pred, scaler=scaler)]

  targets_test_directs, predicts_test_directs = calculate_price_directions(test_targets, test_pred)
  metrics["test_price_direct_accuracy"] = [price_direction_accuracy(targets_test_directs, predicts_test_directs)]

  return metrics.T


class InverseScaleRMSE(RootMeanSquaredError):
	
	def __init__(self, scaler=None, name='root_mean_squared_error', dtype=None):
		super(InverseScaleRMSE, self).__init__(name, dtype)
		self.scaler = scaler
	
	def update_state(self, y_true, y_pred, sample_weight=None):
		y_true_rescaled = self.scaler.inverse_transform(y_true)
		y_pred_rescaled = self.scaler.inverse_transform(y_pred)
    
		return super(InverseScaleRMSE, self).update_state(y_true_rescaled.reshape(-1, ), y_pred_rescaled.reshape(-1, ),
		                                                  sample_weight)
