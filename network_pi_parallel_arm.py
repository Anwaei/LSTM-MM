import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model(units_mlp_x, units_lstm, units_mlp_c):
    input_x = layers.Input(2)
    input_z = layers.Input(None, 1)

    for k in range(len(units_mlp_x)):
        unit = units_mlp_x[k]
        if k == 0:
            out_x = layers.Dense(unit)(input_x)
        else:
            out_x = layers.Dense(unit)(out_x)

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
        out_c = layers.Dense(unit)(out_c)

    # Softmax
    # Model
    # Return
