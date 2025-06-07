import keras


def model_1(vocab_size, embedding_dim, rnn_units):
    embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
    gru = keras.layers.GRU(units=rnn_units, return_sequences=True, return_state=True)(embedding)
    outputs = keras.layers.Dense(units=vocab_size, activation='softmax')(gru)

    return keras.Model(inputs=vocab_size, outputs=outputs)