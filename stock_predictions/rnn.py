def training_stateful_rnn(model, train_data=None, val_data=None, steps_per_epoch=None, callbacks=None, history=None,
                          init_epoch=0, epochs=None):
	if history is None:
		history = {}
	for epoch_num in range(init_epoch, init_epoch + epochs):
		curr_hist = model.fit(train_data, epochs=epoch_num + 1, steps_per_epoch=steps_per_epoch,
		                      validation_data=val_data, shuffle=False, initial_epoch=epoch_num, callbacks=callbacks)
		model.reset_states()
		history = append_to_history(history, curr_hist)
	
	return model, history


def append_to_history(history, to_append):
	if "epoch" not in history.keys():
		history["epoch"] = []
	history["epoch"].extend(to_append.epoch)
	
	for metric in to_append.history.keys():
		if metric not in history.keys():
			history[metric] = []
		history[metric].extend(to_append.history[metric])
	
	return history
