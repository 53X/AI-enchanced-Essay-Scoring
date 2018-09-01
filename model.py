import keras
from keras.layers import Embedding, Conv1D, Input

def essay_model(maxlen, vocab_size):

	input_layer = Input(shape=(maxlen, ), name='raw_essay_input')

	embedding = Embedding(input_dim = vocab_size + 1, output_dim = 300,
						  name='embedding_layer')(input_layer)

	





