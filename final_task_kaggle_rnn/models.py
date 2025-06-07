import keras


def model_1(seq_length, vocab_size, embedding_dim, rnn):
    embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
    gru = rnn(embedding)
    outputs = keras.layers.Dense(units=vocab_size, activation='softmax')(gru)

    return keras.Model(inputs=inputs, outputs=outputs)