import keras
from keras.layers import Embedding, Conv1D, Input, Reshape, Dropout, LSTM, Lambda
from keras import backend as K
from keras.models import Model


def mean_over_time(input_tensor):

	return K.mean(input_tensor, axis=1)


def essay_model(maxlen, vocab_size):

	input_layer = Input(shape=(maxlen, ), name='raw_essay_input')

	embedding = Embedding(input_dim = vocab_size + 1, output_dim = 50,
						  name='embedding_layer')(input_layer)

	reshape = Reshape(target_shape=(1, maxlen, 50), name='reshape_layer')(embedding)					  

	conv_layer = Conv1D(filters=1, kernel_size=3, strides=1, padding='same',
						name='conv_layer', data_format='channels_first', activation='relu')(reshape)

	recurrent_layer = LSTM(units=300, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)(conv_layer)

	mylayer = Lambda(mean_over_time, output_shape=(300, ))(recurrent_layer)

	dropout = Dropout(0.5)(mylayer)

	output = Dense(1, activation='sigmoid')(dropout)

	model = Model(inputs=input_layer, outputs=output)

	model.compile(loss = 'binary_crossentropy', metrics=['binary_accuracy'])

	return model


	






						





