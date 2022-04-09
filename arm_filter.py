import numpy as np
from scipy import stats
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import arm_paras as ap
import arm_model as am
import arm_network_npi_parallel as paranet_npi
import arm_network_pi_parallel as paranet_pi


def flatten(li):
    return sum([[x] if not isinstance(x, list) else flatten(x) for x in li], [])


def one_step_model(rnnmodel):

    model_config = rnnmodel.get_config()
    layers_info = model_config['layers']
    x_layers = []
    s_layers = []
    z_layers = []
    c_layers = []
    f_layers = []

    delkeys = ['return_sequences', 'return_state', 'go_backwards', 'stateful', 'unroll', 'time_major']
    batch_ins = ap.Np

    input_x = layers.Input(shape=ap.nx)
    input_x_ins = tf.convert_to_tensor(np.random.rand(batch_ins, ap.nx))
    input_s = layers.Input(shape=ap.M)
    input_s_ins = tf.convert_to_tensor(np.random.rand(batch_ins, ap.M))
    input_z = layers.Input(shape=ap.nz)
    input_z_ins = tf.convert_to_tensor(np.random.rand(batch_ins, ap.nz))

    for li in layers_info:
        layer_config = li['config']
        if li['class_name'] == 'TimeDistributed':
            if layer_config['name'] == 'Label':
                continue
            else:
                dense_config = layer_config['layer']['config']
                dense_layer = layers.Dense.from_config(config=dense_config)
            if layer_config['name'][0] == 'x':
                x_layers.append(dense_layer)
            elif layer_config['name'][0] == 's':
                s_layers.append(dense_layer)
            elif layer_config['name'][0] == 'c':
                c_layers.append(dense_layer)
            elif layer_config['name'][0] == 'f':
                f_layers.append(dense_layer)
            else:
                raise ValueError('Unexpected dense layer name occurred: ' + layer_config['name'])
        elif li['class_name'] == 'LSTM':
            lstm_config = layer_config
            for key in delkeys:
                del lstm_config[key]
            lstmcell_layer = layers.LSTMCell.from_config(config=lstm_config)
            if layer_config['name'][0] == 'z':
                z_layers.append(lstmcell_layer)
            else:
                raise ValueError('Unexpected lstm layer name occurred: ' + layer_config['name'])
        elif li['class_name'] == 'InputLayer' or li['class_name'] == 'Concatenate':
            pass
        else:
            raise ValueError('Unexpected layer class occurred: ' + li['class_name'])

    output_x = []
    output_s = []
    output_z = []

    for k in range(len(x_layers)):
        if k == 0:
            output_x = x_layers[k](input_x)
        else:
            output_x = x_layers[k](output_x)

    for k in range(len(s_layers)):
        if k == 0:
            output_s = s_layers[k](input_s)
        else:
            output_s = s_layers[k](output_s)

    if len(s_layers) == 0:
        structure = 'parallel'
    else:
        structure = 'integrated'

    input_states = list()
    input_states_ins = list()
    output_states = list()
    for k in range(len(z_layers)):
        input_states.append([layers.Input(shape=z_layers[k].state_size[0]),
                             layers.Input(shape=z_layers[k].state_size[1])])
        input_states_ins.append([tf.convert_to_tensor(np.random.rand(batch_ins, z_layers[k].state_size[0])),
                                 tf.convert_to_tensor(np.random.rand(batch_ins, z_layers[k].state_size[0]))])
    for k in range(len(z_layers)):
        if k == 0:
            output_z, output_state = z_layers[k](input_z, input_states[k])
            output_states.append(output_state)
        else:
            output_z, output_state = z_layers[k](output_z, input_states[k])
            output_states.append(output_state)

    if structure == 'integrated':
        output_c = layers.concatenate([output_x, output_s, output_z])
    else:
        output_c = layers.concatenate([output_x, output_z])

    for k in range(len(c_layers)):
        output_c = c_layers[k](output_c)

    output_final = f_layers[0](output_c)

    input_states = flatten(input_states)
    output_states = flatten(output_states)
    if structure == 'integrated':
        one_step_net = keras.Model(inputs=[input_x, input_s, input_z]+input_states,
                                   outputs=[output_final]+output_states)
    else:
        one_step_net = keras.Model(inputs=[input_x, input_z]+input_states,
                                   outputs=[output_final]+output_states)

    # Instantiate net
    input_states_ins = flatten(input_states_ins)
    if structure == 'integrated':
        _ = one_step_net([input_x_ins, input_s_ins, input_z_ins]+input_states_ins)
    else:
        _ = one_step_net([input_x_ins, input_z_ins] + input_states_ins)

    for li in one_step_net.get_config()['layers']:
        name_new = li['config']['name']
        layer_new = one_step_net.get_layer(name=name_new)
        if li['class_name'] == 'Dense':
            name_origin = name_new[0]+'_wrapped'+name_new[1:]
            layer_origin = rnnmodel.get_layer(name=name_origin)
            layer_new.set_weights(layer_origin.get_weights())
        elif li['class_name'] == 'LSTMCell':
            name_origin = name_new
            layer_origin = rnnmodel.get_layer(name=name_origin)
            layer_new.set_weights(layer_origin.get_weights())
        elif li['class_name'] == 'InputLayer' or li['class_name'] == 'Concatenate':
            pass
        else:
            raise ValueError('Unexpected layer occurred' + li['class_name'])

    return one_step_net, input_states_ins


if __name__ == '__main__':

    T = ap.T
    dt = ap.dt
    K = int(T // dt)
    M = ap.M
    Np = ap.Np

    net_pi_int = keras.models.load_model(ap.net_path_pi_int)
    net_pi_para2 = keras.models.load_model(ap.net_path_pi_para2,
                                           custom_objects={'loss_cce_mode2': paranet_pi.loss_cce_mode2})
    net_pi_para3 = keras.models.load_model(ap.net_path_pi_para3,
                                           custom_objects={'loss_cce_mode3': paranet_pi.loss_cce_mode3})
    net_npi_int = keras.models.load_model(ap.net_path_npi_int)
    net_npi_para2 = keras.models.load_model(ap.net_path_npi_para2,
                                            custom_objects={'loss_cce_mode2': paranet_npi.loss_cce_mode2})
    net_npi_para3 = keras.models.load_model(ap.net_path_npi_para3,
                                            custom_objects={'loss_cce_mode3': paranet_npi.loss_cce_mode3})

    one_step_net, hidden_state_ex = one_step_model(net_npi_int)
    which_net = 'npi_int'

    w = np.zeros(shape=(K, M, Np))
    xp = np.zeros(shape=(K, M, Np, ap.nx))
    mu = np.zeros(shape=(K, M))

    for k in range(K):
        z0 = am.measurement_arm(ap.x0, 0)
        for j in range(M):
            if k == 1:
                for l in range(Np):
                    w[k, j, l] = 1/Np
                    xp[k, j, l, :] = np.random.multivariate_normal(ap.x0, ap.Q0)
                mu[k, j] = 1 if j == ap.s0 else 0
            else:
                pass
