import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import arm_paras as ap
import arm_plot as aplot


def create_model(units_mlp_x, units_mlp_s, units_lstm, units_mlp_c):
    input_x = layers.Input(shape=(None, 2))
    input_s = layers.Input(shape=(None, 3))
    input_z = layers.Input(shape=(None, 1))

    for k in range(len(units_mlp_x)):
        unit = units_mlp_x[k]
        if k == 0:
            out_x = layers.TimeDistributed(layers.Dense(unit, activation='tanh'))(input_x)
        else:
            out_x = layers.TimeDistributed(layers.Dense(unit, activation='tanh'))(out_x)

    for k in range(len(units_mlp_s)):
        unit = units_mlp_s[k]
        if k == 0:
            out_s = layers.TimeDistributed(layers.Dense(unit, activation='tanh'))(input_s)
        else:
            out_s = layers.TimeDistributed(layers.Dense(unit, activation='tanh'))(out_s)

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
                out_z = layers.LSTM(units=unit, return_sequences=True)(out_z)

    out_c = layers.concatenate([out_x, out_s, out_z])

    for k in range(len(units_mlp_c)):
        unit = units_mlp_c[k]
        out_c = layers.TimeDistributed(layers.Dense(unit, activation='sigmoid'))(out_c)

    # out_c = layers.Dense(t_max, activation='sigmoid')(out_c)
    # out_final = layers.Softmax(out_c)
    t_max = ap.T_max_integrated
    out_final = layers.TimeDistributed(layers.Dense(t_max, activation='softmax'))(out_c)

    net = keras.Model(inputs=[input_x, input_s, input_z], outputs=out_final)

    return net


def process_data(data, t_max):
    """
    (1) Read data from npz
    (2) Convert s to one-hot vector
    (3) Convert t to one-hot vector
    (4) Split train and test
    :param data: from npz
    :param t_max: int
    :return: train_input, train_output, test_input, test_output
    """
    x_all, z_all, s_all, t_all, tpm_all, ifreach_all, time_steps_all = aplot.read_data(data)

    s_oh = keras.utils.to_categorical(s_all-1)

    for i in range(t_all.shape[0]):
        for j in range(t_all.shape[2]):
            if t_all[i, 0, j] > t_max:
                t_all[i, 0, j] = t_max
    t_oh = keras.utils.to_categorical(t_all-1)

    input_x = np.swapaxes(x_all, 1, 2)
    input_z = np.swapaxes(z_all, 1, 2)
    input_s = s_oh[:, 0, :, :]
    output_t = t_oh[:, 0, :, :]

    batch_size = input_x.shape[0]
    batch_size_train = int(batch_size * ap.train_prop)

    train_input_x = input_x[0:batch_size_train, :, :]
    train_input_z = input_z[0:batch_size_train, :, :]
    train_input_s = input_s[0:batch_size_train, :, :]
    train_output_t = output_t[0:batch_size_train, :, :]
    train_input = [train_input_x, train_input_s, train_input_z]
    train_output = train_output_t

    test_input_x = input_x[batch_size_train:, :, :]
    test_input_z = input_z[batch_size_train:, :, :]
    test_input_s = input_s[batch_size_train:, :, :]
    test_output_t = output_t[batch_size_train:, :, :]
    test_input = [test_input_x, test_input_s, test_input_z]
    test_output = test_output_t

    return train_input, train_output, test_input, test_output


if __name__ == '__main__':

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    data_path = ap.data_path
    data = np.load(data_path)
    train_input, train_output, test_input, test_output = process_data(data, ap.T_max_integrated)

    units = ap.units_pi_int
    net = create_model(units['mlp_x'], units['mlp_s'], units['lstm'], units['mlp_c'])
    net.compile(optimizer='rmsprop',
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[tf.keras.metrics.CategoricalCrossentropy()])
    net.summary()

    print("========= Start training =========")
    net.fit(x=train_input, y=train_output, epochs=20, batch_size=50)

    print("========= Evaluate =========")
    net.evaluate(test_input, test_output)

    net.save(ap.net_path_pi_int)

    pass
