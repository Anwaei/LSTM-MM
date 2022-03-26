import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import plot_arm as pa


def create_model(units_mlp_x, units_lstm, units_mlp_c, t_max):
    input_x = layers.Input(2)
    input_z = layers.Input(None, 1)

    for k in range(len(units_mlp_x)):
        unit = units_mlp_x[k]
        if k == 0:
            out_x = layers.Dense(unit, activation='tanh')(input_x)
        else:
            out_x = layers.Dense(unit, activation='tanh')(out_x)

    for k in range(len(units_lstm)):
        unit = units_lstm[k]
        if len(units_lstm) == 1:
            out_z = layers.LSTM(units=unit, return_sequences=False)(input_z)
        else:
            if k == 0:
                out_z = layers.LSTM(units=unit, return_sequences=True)(input_z)
            elif k < len(units_lstm) - 1:
                out_z = layers.LSTM(units=unit, return_sequences=True)(out_z)
            else:
                out_z = layers.LSTM(units=unit, return_sequences=False)(out_z)

    out_c = layers.concatenate(out_x, out_z)

    for k in range(len(units_mlp_c)):
        unit = units_mlp_c[k]
        out_c = layers.Dense(unit, activation='sigmoid')(out_c)

    out_c = layers.Dense(t_max, activation='sigmoid')(out_c)
    out_final = layers.Softmax(out_c)

    net = keras.Model(inputs=[input_x, input_z], outputs=out_final)

    return net


def process_data(data):
    """
    (1) Read data from npz
    (2) Construct training/test data of fix length len_data
    (3) Split
    :param data:
    :return:
    """
    x_all, z_all, s_all, t_all, tpm_all, ifreach_all, time_steps_all = pa.read_data(data)
