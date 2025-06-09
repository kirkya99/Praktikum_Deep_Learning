import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import pandas as pd
import numpy as np
import gc
import tensorflow as tf
from tensorflow import keras
from keras import backend as K

def generate_text_with_gru_model(model_url, text, temperature=1.0, num_chars=512):
    # 1. Load your trained model
    model = keras.models.load_model(model_url)

    # get sorted list of all unique characters in the corpus
    vocab = sorted(set(text))
    vocab_size = len(vocab) + 1   # +1 for the OOV token that StringLookup will insert

    # make lookup layers exactly as in your notebook
    ids_from_chars = keras.layers.StringLookup(
        vocabulary=vocab, mask_token=None  # no PAD token
    )
    chars_from_ids = keras.layers.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(), 
        invert=True, 
        mask_token=None
    )

    # helper to turn a string into a vector of IDs
    def text_to_ids(s: str):
        # unicode_split → a TF string Tensor of shape (len(s),)
        chars = tf.strings.unicode_split([s], 'UTF-8')
        return ids_from_chars(chars)

    # helper to turn a list of IDs back into text
    def ids_to_text(ids):
        return tf.strings.reduce_join(chars_from_ids(ids), axis=-1).numpy().astype(str)

    # 3. Seed + pad to the fixed window length your model expects
    seed = "Hey, "
    seq_length = 100   # <— use whatever you trained with

    # convert seed to IDs and pad or trim to length
    seed_ids = text_to_ids(seed).numpy()[0]   # shape (len(seed),)
    seed_ids = seed_ids[-seq_length:]        # keep last seq_length chars
    seed_ids = np.expand_dims(seed_ids, 0)    # make batch of 1
    seed_ids = keras.preprocessing.sequence.pad_sequences(
        seed_ids, maxlen=seq_length, padding='pre'
    )

    # 4. Generate one char at a time
    generated_ids = []

    for _ in range(num_chars):
        # predict next‐char distribution
        preds = model.predict(seed_ids, verbose=0)[0, -1, :]  # (vocab_size,)
        # apply temperature
        preds = np.log(preds + 1e-8) / temperature
        preds = np.exp(preds) / np.sum(np.exp(preds))
        # sample
        next_id = np.random.choice(len(preds), p=preds)
        generated_ids.append(next_id)
        # shift window and append
        seed_ids = np.roll(seed_ids, -1, axis=1)
        seed_ids[0, -1] = next_id

    # 5. Decode back to a string
    gen_text = ids_to_text(np.array([generated_ids]))
    print(seed)

    for text in gen_text:
        print(text)

    del model
    K.clear_session()
    gc.collect()

def generate_text_with_lstm_model(model_url, text, temperature=1.0, num_chars=512):
    # 1. Load your trained LSTM model
    model = keras.models.load_model(model_url)

    # 2. Rebuild your char‐level vocabulary/lookups exactly as in training
    vocab = sorted(set(text))
    vocab_size = len(vocab) + 1   # +1 for any OOV token

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

    # 3. Prepare your seed and pad/trim to the sequence length you trained with
    seed = "Hey, this is my LSTM: "
    seq_length = 100   # ← must match the seq length you used during training

    seed_ids = text_to_ids(seed).numpy()[0]      # shape (len(seed),)
    seed_ids = seed_ids[-seq_length:]           # keep the last seq_length tokens
    seed_ids = np.expand_dims(seed_ids, 0)       # batch size 1
    seed_ids = keras.preprocessing.sequence.pad_sequences(
        seed_ids, maxlen=seq_length, padding='pre'
    )

    # 4. Sampling loop
    generated_ids = []

    for _ in range(num_chars):
        # model.predict will return logits over the vocab
        logits = model.predict(seed_ids, verbose=0)[0, -1, :]  # shape (vocab_size,)
        # apply temperature
        logits = np.log(logits + 1e-8) / temperature
        probs = np.exp(logits) / np.sum(np.exp(logits))
        # draw one character ID
        next_id = np.random.choice(len(probs), p=probs)
        generated_ids.append(next_id)
        # slide the window one step and append the new ID
        seed_ids = np.roll(seed_ids, -1, axis=1)
        seed_ids[0, -1] = next_id

    # 5. Decode and print
    gen_text = ids_to_text(np.array([generated_ids]))
    print("Seed:", seed)
    print("Generated continuation:\n")
    for t in gen_text:
        print(t)

    del model
    K.clear_session()
    gc.collect()