#!/usr/bin/env python3
"""Creating an autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creating an autoencoder
    input_dims is an integer containing the
        dimensions of the model input
    hidden_layers is a list containing the number of
        nodes for each hidden layer in the encoder, respectively
    the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the
        dimensions of the latent space representation

    Returns: encoder, decoder, auto

    """
    k = keras.layers
    input = keras.Input(shape=(input_dims,))
    encoded_layer = k.Dense(hidden_layers[0], activation='relu')(input)
    for layer in hidden_layers[1:]:
        encoded_layer = k.Dense(layer, activation='relu')(encoded_layer)
    encoded_layer = k.Dense(latent_dims, activation='relu')(encoded_layer)
    encoder = keras.Model(input, encoded_layer)

    coded_input = keras.Input(shape=(latent_dims,))
    decoded_layer = k.Dense(hidden_layers[-1], activation='relu')(coded_input)
    for dim in hidden_layers[-2::-1]:
        decoded_layer = k.Dense(dim, activation='relu')(decoded_layer)
    decoded_layer = k.Dense(input_dims, activation='sigmoid')(decoded_layer)
    decoder = keras.Model(coded_input, decoded_layer)

    auto = keras.Model(input, decoder(encoder(input)))
    auto.compile(loss='binary_crossentropy', optimizer='adam')

    return encoder, decoder, auto