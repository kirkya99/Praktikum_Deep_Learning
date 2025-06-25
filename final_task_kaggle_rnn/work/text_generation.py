import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import pandas as pd
import numpy as np
import gc
import tensorflow as tf
from tensorflow import keras
from keras import backend as K


def generate_text_with_gru_model(model_url, text, temperature=1.0, num_chars=512):
    """
    Generate text using a pre-trained GRU model.

    Parameters
    ----------
    model_url : str
        URL to the pre-trained GRU model.
    text : str
        The text corpus to use for generating text.
    temperature : float, optional (default=1.0)
        The temperature to use for generating text. A temperature close to 0 will result in more predictable text, 
        while a temperature close to 1 will result in more random text.
    num_chars : int, optional (default=512)
        The number of characters to generate.

    Returns
    -------
    None
    """
    model = keras.models.load_model(model_url)

    vocab = sorted(set(text))
    vocab_size = len(vocab) + 1 

    ids_from_chars = keras.layers.StringLookup(
        vocabulary=vocab, mask_token=None
    )
    chars_from_ids = keras.layers.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(), 
        invert=True, 
        mask_token=None
    )

    def text_to_ids(s: str):
        chars = tf.strings.unicode_split([s], 'UTF-8')
        return ids_from_chars(chars)

    def ids_to_text(ids):
        return tf.strings.reduce_join(chars_from_ids(ids), axis=-1).numpy().astype(str)

    seed = "To be "
    seq_length = 100

    seed_ids = text_to_ids(seed).numpy()[0]
    seed_ids = seed_ids[-seq_length:]     
    seed_ids = np.expand_dims(seed_ids, 0)
    seed_ids = keras.preprocessing.sequence.pad_sequences(
        seed_ids, maxlen=seq_length, padding='pre'
    )

    generated_ids = []

    for _ in range(num_chars):
        preds = model.predict(seed_ids, verbose=0)[0, -1, :]
        preds = np.log(preds + 1e-8) / temperature
        preds = np.exp(preds) / np.sum(np.exp(preds))
        next_id = np.random.choice(len(preds), p=preds)
        generated_ids.append(next_id)
        seed_ids = np.roll(seed_ids, -1, axis=1)
        seed_ids[0, -1] = next_id

    gen_text = ids_to_text(np.array([generated_ids]))
    print(seed)

    for text in gen_text:
        print(text)

    del model
    K.clear_session()
    gc.collect()

def generate_text_with_lstm_model(model_url, text, temperature=1.0, num_chars=512):
    """
    Generate text using a pre-trained LSTM model.

    Parameters
    ----------
    model_url : str
        URL of the pre-trained LSTM model.
    text : str
        The text to be used for generating the vocabulary.
    temperature : float, optional (default=1.0)
        The temperature to use for generating text. A temperature close to 0 will result in more predictable text, 
        while a temperature close to 1 will result in more random text.
    num_chars : int, optional (default=512)
        The number of characters to generate.

    Returns
    -------
    None
    """
    model = keras.models.load_model(model_url)

    vocab = sorted(set(text))
    vocab_size = len(vocab) + 1

    ids_from_chars = keras.layers.StringLookup(
        vocabulary=vocab, mask_token=None
    )
    chars_from_ids = keras.layers.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(),
        invert=True,
        mask_token=None
    )

    def text_to_ids(s: str):
        chars = tf.strings.unicode_split([s], 'UTF-8')
        return ids_from_chars(chars)

    def ids_to_text(ids):
        return tf.strings.reduce_join(chars_from_ids(ids), axis=-1).numpy().astype(str)

    seed = "To be "
    seq_length = 100

    seed_ids = text_to_ids(seed).numpy()[0]
    seed_ids = seed_ids[-seq_length:]
    seed_ids = np.expand_dims(seed_ids, 0)
    seed_ids = keras.preprocessing.sequence.pad_sequences(
        seed_ids, maxlen=seq_length, padding='pre'
    )

    generated_ids = []

    for _ in range(num_chars):
        logits = model.predict(seed_ids, verbose=0)[0, -1, :]
        logits = np.log(logits + 1e-8) / temperature
        probs = np.exp(logits) / np.sum(np.exp(logits))
        next_id = np.random.choice(len(probs), p=probs)
        generated_ids.append(next_id)
        seed_ids = np.roll(seed_ids, -1, axis=1)
        seed_ids[0, -1] = next_id

    gen_text = ids_to_text(np.array([generated_ids]))
    print("Seed:", seed)
    print("Generated continuation:\n")
    for t in gen_text:
        print(t)

    del model
    K.clear_session()
    gc.collect()