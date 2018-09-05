import keras
from keras.layers import Embedding, Conv1D, Input, Reshape, Dropout, LSTM


def essay_model(maxlen, vocab_size):

	input_layer = Input(shape=(maxlen, ), name='raw_essay_input')

	embedding = Embedding(input_dim = vocab_size + 1, output_dim = 50,
						  name='embedding_layer')(input_layer)

	reshape = Reshape(target_shape=(1, maxlen, 50), name='reshape_layer')(embedding)					  

	conv_layer = Conv1D(filters=1, kernel_size=3, strides=1, padding='same',
						name='conv_layer', data_format='channels_first', activation='relu')(reshape)

	recurrent_layer = LSTM()

						





