import keras

def get_gru_model_1(vocab_size, embedding_dim, rnn_units):
    inputs = keras.layers.Input(shape=(None,), dtype='int32', name='input_tokens')
    embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
    gru, gru_state = keras.layers.GRU(units=rnn_units, return_sequences=True, return_state=True)(embedding)
    gru, gru_state = keras.layers.GRU(units=rnn_units, return_sequences=True, return_state=True)(gru)
    outputs = keras.layers.Dense(units=vocab_size, activation='softmax')(gru)

    return keras.Model(inputs=inputs, outputs=outputs)

def get_gru_model_2(vocab_size, embedding_dim, rnn_units):
    inputs = keras.layers.Input(shape=(None,), dtype='int32', name='input_tokens')
    embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
    gru, gru_state = keras.layers.GRU(units=rnn_units, return_sequences=True, return_state=True)(embedding)
    dropout = keras.layers.Dropout(0.2)(gru)
    gru, gru_state = keras.layers.GRU(units=rnn_units, return_sequences=True, return_state=True)(dropout)
    outputs = keras.layers.Dense(units=vocab_size, activation='softmax')(gru)

    return keras.Model(inputs=inputs, outputs=outputs)

def get_lstm_model_1(vocab_size, embedding_dim, rnn_units):
    inputs = keras.layers.Input(shape=(None,), dtype='int32', name='input_tokens')
    embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
    gru, gru_state = keras.layers.LSTM(units=rnn_units, return_sequences=True, return_state=True)(embedding)
    gru, gru_state = keras.layers.LSTM(units=rnn_units, return_sequences=True, return_state=True)(gru)
    outputs = keras.layers.Dense(units=vocab_size, activation='softmax')(gru)

    return keras.Model(inputs=inputs, outputs=outputs)

def get_lstm_model_2(vocab_size, embedding_dim, rnn_units):
    inputs = keras.layers.Input(shape=(None,), dtype='int32', name='input_tokens')
    embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
    gru, gru_state = keras.layers.LSTM(units=rnn_units, return_sequences=True, return_state=True)(embedding)
    dropout = keras.layers.Dropout(0.2)(gru)
    gru, gru_state = keras.layers.LSTM(units=rnn_units, return_sequences=True, return_state=True)(dropout)
    outputs = keras.layers.Dense(units=vocab_size, activation='softmax')(gru)

    return keras.Model(inputs=inputs, outputs=outputs)