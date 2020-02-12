from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM

ENC_LAYER_1_UNITS = 64
ENC_LAYER_2_UNITS = 32


class LSTMEncoder(Model):
	def __init__(self, interm_dim, name=None, **kwargs):
		super(LSTMEncoder, self).__init__(name=name, **kwargs)
		self.encoder_layer_1 = LSTM(units=ENC_LAYER_1_UNITS, return_sequences=True, name="enc_layer_1")
		self.encoder_layer_2 = LSTM(units=ENC_LAYER_2_UNITS, return_sequences=True, name="enc_layer_2")
		self.encoder_output_layer = LSTM(units=interm_dim, return_sequences=True, name="enc_output_layer")
	
	def build(self, input_shape):
		return super().build(input_shape)
	
	def compute_output_shape(self, input_shape):
		return super().compute_output_shape(input_shape)
	
	def call(self, inputs):
		encoded = self.encoder_layer_1(inputs)
		encoded = self.encoder_layer_2(encoded)
		output = self.encoder_output_layer(encoded)
		
		return output


class LSTMDecoder(Model):
	def __init__(self, orig_dim, name=None, **kwargs):
		super(LSTMDecoder, self).__init__(name=name, **kwargs)
		self.decoder_layer_1 = LSTM(units=ENC_LAYER_2_UNITS, return_sequences=True, name="dec_layer_1")
		self.decoder_layer_2 = LSTM(units=ENC_LAYER_1_UNITS, return_sequences=True, name="dec_layer_2")
		self.decoder_output_layer = LSTM(units=orig_dim, return_sequences=True, name="dec_output_layer")
	
	def call(self, inputs):
		decoded = self.decoder_layer_1(inputs)
		decoded = self.decoder_layer_2(decoded)
		outputs = self.decoder_output_layer(decoded)
		
		return outputs


class LSTMAutoEncoder(Model):
	def __init__(self, interm_dim, orig_dim, name=None):
		super(LSTMAutoEncoder, self).__init__(name=name)
		self.encoder = LSTMEncoder(interm_dim)
		self.decoder = LSTMDecoder(orig_dim)
	
	def call(self, inputs):
		interm = self.encoder(inputs)
		outputs = self.decoder(interm)
		return outputs
