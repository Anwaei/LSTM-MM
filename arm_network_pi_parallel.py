import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import arm_paras as ap
import arm_plot as aplot


def create_model(units_mlp_x, units_lstm, units_mlp_c):
    input_x = layers.Input(shape=(None, 2))
    input_s = layers.Input(shape=(None, 3))
    input_z = layers.Input(shape=(None, 1))

    for k in range(len(units_mlp_x)):
        unit = units_mlp_x[k]
        if k == 0:
            out_x = layers.TimeDistributed(layers.Dense(unit, activation='tanh',
                                                        name='x_dense' + str(k)),
                                           name='x_wrapped_dense' + str(k))(input_x)
        else:
            out_x = layers.TimeDistributed(layers.Dense(unit, activation='tanh',
                                                        name='x_dense' + str(k)),
                                           name='x_wrapped_dense' + str(k))(out_x)

    for k in range(len(units_lstm)):
        unit = units_lstm[k]
        if len(units_lstm) == 1:
            out_z = layers.LSTM(units=unit, return_sequences=False, name='z_lstm' + str(k))(input_z)
        else:
            if k == 0:
                out_z = layers.LSTM(units=unit, return_sequences=True, name='z_lstm' + str(k))(input_z)
            else:
                out_z = layers.LSTM(units=unit, return_sequences=True, name='z_lstm' + str(k))(out_z)

    out_c = layers.concatenate([out_x, out_z])

    t_max = ap.T_max_integrated

    for k in range(len(units_mlp_c)):
        unit = units_mlp_c[k]
        out_c = layers.TimeDistributed(layers.Dense(unit, activation='sigmoid',
                                                    name='c_dense' + str(k)),
                                       name='c_wrapped_dense' + str(k))(out_c)
    out_c = layers.TimeDistributed(layers.Dense(t_max, activation='softmax', name='f_dense'),
                                   name='f_wrapped_dense')(out_c)

    label_s = layers.TimeDistributed(layers.Activation('linear'), name='Label')(input_s)
    out_final = layers.concatenate([out_c, label_s])

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
    train_output = np.concatenate((train_output_t, train_input_s), axis=-1)

    test_input_x = input_x[batch_size_train:, :, :]
    test_input_z = input_z[batch_size_train:, :, :]
    test_input_s = input_s[batch_size_train:, :, :]
    test_output_t = output_t[batch_size_train:, :, :]
    test_input = [test_input_x, test_input_s, test_input_z]
    test_output = np.concatenate((test_output_t, test_input_s), axis=-1)

    return train_input, train_output, test_input, test_output


def loss_cce_mode1(y_true, y_pred):
    cce = keras.losses.categorical_crossentropy(y_true[:, :, 0:-3], y_pred[:, :, 0:-3])
    selected_cce = tf.multiply(y_true[:, :, -3], cce)
    num = tf.reduce_sum(y_true[:, :, -3])
    loss = tf.reduce_sum(selected_cce)/num
    return loss


def loss_cce_mode2(y_true, y_pred):
    cce = keras.losses.categorical_crossentropy(y_true[:, :, 0:-3], y_pred[:, :, 0:-3])
    selected_cce = tf.multiply(y_true[:, :, -2], cce)
    num = tf.reduce_sum(y_true[:, :, -2])
    loss = tf.reduce_sum(selected_cce)/num
    return loss


def loss_cce_mode3(y_true, y_pred):
    cce = keras.losses.categorical_crossentropy(y_true[:, :, 0:-3], y_pred[:, :, 0:-3])
    selected_cce = tf.multiply(y_true[:, :, -1], cce)
    num = tf.reduce_sum(y_true[:, :, -1])
    loss = tf.reduce_sum(selected_cce)/num
    return loss


if __name__ == '__main__':

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    data_path = ap.data_path
    data = np.load(data_path)

    train_input_1, train_output_1, test_input_1, test_output_1 = process_data(data, ap.T_max_parallel[0])
    train_input_2, train_output_2, test_input_2, test_output_2 = process_data(data, ap.T_max_parallel[1])
    train_input_3, train_output_3, test_input_3, test_output_3 = process_data(data, ap.T_max_parallel[2])

    units1 = ap.units_pi_para1
    net1 = create_model(units1['mlp_x'], units1['lstm'], units1['mlp_c'])
    net1.compile(optimizer='rmsprop',
                 loss=loss_cce_mode1,
                 metrics=loss_cce_mode1)
    net1.summary()

    print("========= Start training net1 =========")
    net1.fit(x=train_input_1, y=train_output_1, epochs=5, batch_size=ap.bs)
    print("========= Evaluate net1 =========")
    net1.evaluate(test_input_1, test_output_1)

    units2 = ap.units_pi_para2
    net2 = create_model(units2['mlp_x'], units2['lstm'], units2['mlp_c'])
    net2.compile(optimizer='rmsprop',
                 loss=loss_cce_mode2,
                 metrics=loss_cce_mode2)
    net2.summary()

    print("========= Start training net2 =========")
    net2.fit(x=train_input_2, y=train_output_2, epochs=5, batch_size=ap.bs)
    print("========= Evaluate net2 =========")
    net2.evaluate(test_input_2, test_output_2)

    units3 = ap.units_pi_para3
    net3 = create_model(units3['mlp_x'], units3['lstm'], units3['mlp_c'])
    net3.compile(optimizer='rmsprop',
                 loss=loss_cce_mode3,
                 metrics=loss_cce_mode3)
    net3.summary()

    print("========= Start training net3 =========")
    net3.fit(x=train_input_3, y=train_output_2, epochs=5, batch_size=ap.bs)
    print("========= Evaluate net3 =========")
    net3.evaluate(test_input_2, test_output_2)

    net2.save(ap.net_path_pi_para2)
    net3.save(ap.net_path_pi_para3)
