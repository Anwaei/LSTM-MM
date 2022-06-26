import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import tracking_paras as tkp
import tracking_model as tkm
import tracking_plot as tkplot
import tracking_network_npi_parallel as paranet_npi
import tracking_network_pi_parallel as paranet_pi
import time


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
    batch_ins = tkp.Np

    input_x = layers.Input(shape=tkp.nx)
    input_x_ins = tf.convert_to_tensor(np.random.rand(batch_ins, tkp.nx))
    input_s = layers.Input(shape=tkp.M)
    input_s_ins = tf.convert_to_tensor(np.random.rand(batch_ins, tkp.M))
    input_z = layers.Input(shape=tkp.nz)
    input_z_ins = tf.convert_to_tensor(np.random.rand(batch_ins, tkp.nz))

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


def compute_cmtp(nets, which_net, x, z, s, hidden):
    x = tf.convert_to_tensor(x)
    z = tf.tile(z[None, :], (tkp.Np, 1))

    if which_net == 'npi_int' or which_net == 'pi_int':
        net_int = nets[0]
        s_cat = tf.tile(keras.utils.to_categorical(s, tkp.M)[None, :], (tkp.Np, 1))
        out = net_int([x, s_cat, z] + hidden)
        hidden_new = out[1:]
        if which_net == 'pi_int':
            soutdtr = out[0]
            cmtp = np.zeros(shape=(M, tkp.Np))
            for l in range(tkp.Np):
                for t in range(1, tkp.T_max_integrated + 1):
                    tpm = tkm.tpm_tracking(x[l, :], t)
                    tp = tpm[s, :]
                    cmtp[:, l] = cmtp[:, l] + tp * soutdtr[l, t - 1]
                cmtp[:, l] = cmtp[:, l] / np.sum(cmtp[:, l])
        elif which_net == 'npi_int':
            cmtp = np.transpose(out[0])
        else:
            raise ValueError('Error net type')
    elif which_net == 'npi_para' or which_net == 'pi_para':
        net_para = nets[s]
        out = net_para([x, z] + hidden)
        hidden_new = out[1:]
        if which_net == 'pi_para':
            soutdtr = out[0]
            cmtp = np.zeros(shape=(M, tkp.Np))
            for l in range(tkp.Np):
                for t in range(1, tkp.T_max_parallel[s] + 1):
                    tpm = tkm.tpm_tracking(x[l, :], t)
                    tp = tpm[s, :]
                    cmtp[:, l] = cmtp[:, l] + tp * soutdtr[l, t - 1]
                cmtp[:, l] = cmtp[:, l] / np.sum(cmtp[:, l])
        elif which_net == 'npi_para':
            cmtp = np.transpose(out[0])
        else:
            raise ValueError('Error net type')
    else:
        raise ValueError('Error net type')

    return cmtp, hidden_new


def compute_zcpredict_likelihood(x_pre, z, s):
    # t1=time.clock()
    lam = tkm.dynamic_tracking(sc=s + 1, x_p=x_pre, q=np.zeros(tkp.nq))
    # zli = am.compute_meas_likelihood(x=lam, z=z, s=s)
    # cli = am.compute_constraint_likelihood(x=lam)
    # li = zli*cli
    # t2 = time.clock()
    zlogli = tkm.compute_meas_loglikelihood(x=lam, z=z, s=s)
    clogli = tkm.compute_constraint_loglikelihood(x=lam)
    li = np.exp(zlogli + clogli)
    # t3 = time.clock()
    return li


def compute_zc_likelihood(x, z, s):
    # zli = am.compute_meas_likelihood(x=x, z=z, s=s)
    # cli = am.compute_constraint_likelihood(x=x)
    # li = zli*cli
    zlogli = tkm.compute_meas_loglikelihood(x=x, z=z, s=s)
    clogli = tkm.compute_constraint_loglikelihood(x=x)
    li = np.exp(zlogli + clogli)
    return li


def sample_auxiliary_variables(v):
    M = tkp.M
    Np = tkp.Np
    if v.shape != (M, Np):
        raise ValueError('Incorrect probability dimension for v')
    v_flatten = np.reshape(v, (M * Np))
    index = np.random.choice(a=M * Np, size=Np, p=v_flatten)
    xi = index // Np
    xi.astype(int)
    zeta = index - xi * Np
    zeta.astype(int)
    return xi, zeta


if __name__ == '__main__':

    which_net = 'npi_int'
    print(which_net)

    # ---------------------------------------------------------------------- #
    # -----------------------------| S T A R T |---------------------------- #
    # ---------------------------------------------------------------------- #
    T = tkp.T
    dt = tkp.dt
    K = int(T / dt)
    M = tkp.M
    Np = tkp.Np
    run_batch = tkp.run_batch

    nets = list()
    # hiddens_ex = list()
    if which_net == 'pi_int':
        net_pi_int = keras.models.load_model(tkp.net_path_pi_int)
        one_step_net, hidden_ex = one_step_model(net_pi_int)
        nets.append(one_step_net)
    elif which_net == 'pi_para':
        net_pi_para1 = keras.models.load_model(tkp.net_path_pi_para1,
                                               custom_objects={'loss_cce_mode1': paranet_pi.loss_cce_mode1})
        net_pi_para2 = keras.models.load_model(tkp.net_path_pi_para2,
                                               custom_objects={'loss_cce_mode2': paranet_pi.loss_cce_mode2})
        net_pi_para3 = keras.models.load_model(tkp.net_path_pi_para3,
                                               custom_objects={'loss_cce_mode3': paranet_pi.loss_cce_mode3})
        net_pi_para4 = keras.models.load_model(tkp.net_path_pi_para4,
                                               custom_objects={'loss_cce_mode4': paranet_pi.loss_cce_mode4})
        net_pi_para5 = keras.models.load_model(tkp.net_path_pi_para5,
                                               custom_objects={'loss_cce_mode5': paranet_pi.loss_cce_mode5})
        one_step_net1, hidden_ex1 = one_step_model(net_pi_para1)
        one_step_net2, hidden_ex2 = one_step_model(net_pi_para2)
        one_step_net3, hidden_ex3 = one_step_model(net_pi_para3)
        one_step_net4, hidden_ex4 = one_step_model(net_pi_para4)
        one_step_net5, hidden_ex5 = one_step_model(net_pi_para5)
        nets.extend([one_step_net1, one_step_net2, one_step_net3, one_step_net4, one_step_net5])
        hidden_ex = [hidden_ex1, hidden_ex2, hidden_ex3, hidden_ex4, hidden_ex5]
    elif which_net == 'npi_int':
        net_npi_int = keras.models.load_model(tkp.net_path_npi_int)
        one_step_net, hidden_ex = one_step_model(net_npi_int)
        nets.append(one_step_net)
    elif which_net == 'npi_para':
        net_npi_para1 = keras.models.load_model(tkp.net_path_npi_para1,
                                                custom_objects={'loss_cce_mode1': paranet_npi.loss_cce_mode1})
        net_npi_para2 = keras.models.load_model(tkp.net_path_npi_para2,
                                                custom_objects={'loss_cce_mode2': paranet_npi.loss_cce_mode2})
        net_npi_para3 = keras.models.load_model(tkp.net_path_npi_para3,
                                                custom_objects={'loss_cce_mode3': paranet_npi.loss_cce_mode3})
        net_npi_para4 = keras.models.load_model(tkp.net_path_npi_para4,
                                                custom_objects={'loss_cce_mode4': paranet_npi.loss_cce_mode4})
        net_npi_para5 = keras.models.load_model(tkp.net_path_npi_para5,
                                                custom_objects={'loss_cce_mode5': paranet_npi.loss_cce_mode5})
        one_step_net1, hidden_ex1 = one_step_model(net_npi_para1)
        one_step_net2, hidden_ex2 = one_step_model(net_npi_para2)
        one_step_net3, hidden_ex3 = one_step_model(net_npi_para3)
        one_step_net4, hidden_ex4 = one_step_model(net_npi_para4)
        one_step_net5, hidden_ex5 = one_step_model(net_npi_para5)
        nets.extend([one_step_net1, one_step_net2, one_step_net3, one_step_net4, one_step_net5])
        hidden_ex = [hidden_ex1, hidden_ex2, hidden_ex3, hidden_ex4, hidden_ex5]
    else:
        raise ValueError('Error net structure')

    x0 = tkp.x0
    s0 = tkp.s0
    z0 = tkm.measurement_tracking(x0, 0)

    xtrue_all = np.zeros(shape=(run_batch, K + 1, tkp.nx))
    strue_all = np.zeros(shape=(run_batch, K + 1))
    xest_all = np.zeros(shape=(run_batch, K + 1, tkp.nx))
    xp_all = np.zeros(shape=(run_batch, K + 1, M, Np, tkp.nx))
    w_all = np.zeros(shape=(run_batch, K + 1, M, Np))
    mu_all = np.zeros(shape=(run_batch, K + 1, M))
    z_all = np.zeros(shape=(run_batch, K + 1, tkp.nz))
    cmtp_all = np.zeros(shape=(run_batch, K + 1, M, M, Np))
    what_all = np.zeros(shape=(run_batch, K + 1, M, M, Np))
    what_sum_all = np.zeros(shape=(run_batch, K + 1, M))
    zcliprd_all = np.zeros(shape=(run_batch, K + 1, M, M, Np))
    v_all = np.zeros(shape=(run_batch, K + 1, M, M, Np))
    xi_all = np.zeros(shape=(run_batch, K + 1, M, Np), dtype='int')
    zeta_all = np.zeros(shape=(run_batch, K + 1, M, Np), dtype='int')
    q_proposal_all = np.random.multivariate_normal(mean=np.zeros(tkp.nq), cov=tkp.Q, size=(K + 1, M, Np))

    data = np.load(tkp.data_path)
    x_data, z_data, s_data, t_data, tpm_data, ifreach_data, time_steps_data = tkplot.read_data(data)
    size_run = int(x_data.shape[0] * tkp.train_prop)
    size_run = int(x_data.shape[0] * tkp.train_prop) + 200
    size_run = 600
    size_run = tkp.size_run
    # xtrue_batch = np.swapaxes(x_data[size_run:, :, :], 1, 2)
    # ztrue_batch = np.swapaxes(z_data[size_run:, :, :], 1, 2)
    # strue_batch = s_data[size_run:, 0, :]
    # ttrue_batch = t_data[size_run:, 0, :]
    # ifreach_batch = ifreach_data[size_run:, 0, :]
    # time_steps_batch = time_steps_data[size_run:, 0, :]
    # time_steps = time_steps_batch[0, 0:K]

    xtrue_batch = np.swapaxes(x_data[tkp.sr, :, :], 1, 2)
    ztrue_batch = np.swapaxes(z_data[tkp.sr, :, :], 1, 2)
    strue_batch = s_data[tkp.sr, 0, :]
    ttrue_batch = t_data[tkp.sr, 0, :]
    time_steps_batch = time_steps_data[tkp.sr, 0, :]
    time_steps = time_steps_batch[0, 0:K]


    for n in tqdm(range(run_batch)):
        hidden_int_all = []
        hidden_para_all = [[]]

        xtrue_all[n, 0, :] = x0
        xtrue_all[n, 1:, :] = xtrue_batch[n, 0:K, :]
        z_all[n, 0, :] = z0
        z_all[n, 1:, :] = ztrue_batch[n, 0:K, :]
        strue_all[n, 0] = s0
        strue_all[n, 1:] = strue_batch[n, 0:K]

        # k=0:
        for j in range(M):
            for l in range(Np):
                w_all[n, 0, j, l] = 1 / Np
                xp_all[n, 0, j, l, :] = np.random.multivariate_normal(x0, tkp.Q0)
            mu_all[n, 0, j] = 0.96 if j == s0-1 else 0.01
        if which_net == 'pi_int' or which_net == 'npi_int':
            hidden0 = list()
            for l in range(len(hidden_ex)):
                hidden0.append(tf.convert_to_tensor(np.zeros(shape=hidden_ex[l].shape)))
            hidden_int_all.append(hidden0)
        elif which_net == 'pi_para' or which_net == 'npi_para':
            for j in range(M):
                hidden0 = list()
                for l in range(len(hidden_ex[j])):
                    hidden0.append(tf.convert_to_tensor(np.zeros(shape=hidden_ex[j][l].shape)))
                hidden_para_all[0].append(hidden0)

        for k in range(1, K + 1):
            zcli = np.zeros(shape=(M, Np))
            z_pre = z_all[n, k - 1, :]
            z = z_all[n, k, :]
            for i in range(M):
                xp_pre = xp_all[n, k - 1, i, :, :]
                if which_net == 'pi_int' or which_net == 'npi_int':
                    hidden_pre = hidden_int_all[k - 1]
                elif which_net == 'pi_para' or which_net == 'npi_para':
                    hidden_pre = hidden_para_all[k - 1][i]
                else:
                    raise ValueError('Error net structure')
                cmtp_pre, hidden_new = compute_cmtp(nets=nets, which_net=which_net,
                                                    x=xp_pre, z=z_pre, s=i,
                                                    hidden=hidden_pre)
                cmtp_all[n, k - 1, i, :, :] = cmtp_pre
                if which_net == 'pi_int' or which_net == 'npi_int':
                    if i == 0:
                        hidden_int_all.append(hidden_new)
                elif which_net == 'pi_para' or which_net == 'npi_para':
                    if i == 0:
                        hidden_para_all.append([])
                    hidden_para_all[k].append(hidden_new)
            # print(1)
            for j in range(M):
                for i in range(M):
                    for l in range(Np):
                        # what_pre = cmtp_all[n, k - 1, i, j, l] * mu_all[n, k - 1, i] * w_all[n, k - 1, i, l]
                        # what_all[n, k - 1, i, j, l] = what_pre
                        what_all[n, k - 1, i, j, l] = cmtp_all[n, k - 1, i, j, l] * mu_all[n, k - 1, i] * w_all[n, k - 1, i, l]
                what_pre_sum = np.sum(what_all[n, k - 1, :, j, :])
                what_sum_all[n, k - 1, j] = what_pre_sum
                what_all[n, k - 1, :, j, :] = what_all[n, k - 1, :, j, :] / what_pre_sum
            # print(2)
            v_pre_sum = np.zeros(M)
            for j in range(M):
                for i in range(M):
                    for l in range(Np):
                        # zcliprd_pre = compute_zcpredict_likelihood(x_pre=xp_all[n, k - 1, i, l, :], z=z, s=j)
                        # zcliprd_all[n, k - 1, i, j, l] = zcliprd_pre
                        # v_pre = zcliprd_pre * what_all[n, k - 1, i, j, l]
                        # v_all[n, k - 1, i, j, l] = v_pre
                        v_all[n, k - 1, i, j, l] = compute_zcpredict_likelihood(x_pre=xp_all[n, k - 1, i, l, :], z=z, s=j) \
                                                * what_all[n, k - 1, i, j, l]
                v_pre_sum[j] = np.sum(v_all[n, k - 1, :, j, :])
                v_all[n, k - 1, :, j, :] = v_all[n, k - 1, :, j, :] / v_pre_sum[j]
            # print(3)
            for j in range(M):
                xi_all[n, k - 1, j, :], zeta_all[n, k - 1, j, :] = sample_auxiliary_variables(v_all[n, k - 1, :, j, :])
                for l in range(Np):
                    xi = xi_all[n, k - 1, j, l]
                    zeta = zeta_all[n, k - 1, j, l]
                    xp = tkm.dynamic_tracking(sc=j + 1, x_p=xp_all[n, k - 1, xi, zeta, :], q=q_proposal_all[k - 1, j, l, :])
                    xp_all[n, k, j, l, :] = xp
                    zcli[j, l] = compute_zc_likelihood(x=xp, z=z, s=j)
                    w_all[n, k, j, l] = zcli[j, l] * what_all[n, k - 1, xi, j, zeta] / v_all[n, k - 1, xi, j, zeta]
                w_all[n, k, j, :] = w_all[n, k, j, :] / np.sum(w_all[n, k, j, :])
            # print(4)
            for j in range(M):
                li = np.sum(zcli[j, :]) / Np
                mu = li * what_sum_all[n, k - 1, j]
                # mu = v_pre_sum[j] * what_sum_all[n, k - 1, j]
                # for l2 in range(Np):
                #     mu = mu + w_all[n, k, j, l2] * what_sum_all[n, k - 1, j]
                mu_all[n, k, j] = mu
            mu_all[n, k, :] = mu_all[n, k, :] / np.sum(mu_all[n, k, :])
            # print(5)
            xest = np.zeros(tkp.nx)
            for j in range(M):
                xestj = np.zeros(tkp.nx)
                for l in range(Np):
                    xestj = xestj + w_all[n, k, j, l] * xp_all[n, k, j, l, :]
                xest = xest + mu_all[n, k, j] * xestj
            xest_all[n, k, :] = xest
            # print(6)

    np.savez(file=tkp.filter_data_path + '_' + which_net + '.npz',
             xtrue_all=xtrue_all,
             strue_all=strue_all,
             xest_all=xest_all,
             mu_all=mu_all,
             z_all=z_all,
             time_steps=time_steps)

    # ---------------------------------------------------------------------- #
    # -----------------------------|  E  N  D  |---------------------------- #
    # ---------------------------------------------------------------------- #


    which_net = 'npi_para'
    print(which_net)

    # ---------------------------------------------------------------------- #
    # -----------------------------| S T A R T |---------------------------- #
    # ---------------------------------------------------------------------- #
    T = tkp.T
    dt = tkp.dt
    K = int(T / dt)
    M = tkp.M
    Np = tkp.Np
    run_batch = tkp.run_batch

    nets = list()
    # hiddens_ex = list()
    if which_net == 'pi_int':
        net_pi_int = keras.models.load_model(tkp.net_path_pi_int)
        one_step_net, hidden_ex = one_step_model(net_pi_int)
        nets.append(one_step_net)
    elif which_net == 'pi_para':
        net_pi_para1 = keras.models.load_model(tkp.net_path_pi_para1,
                                               custom_objects={'loss_cce_mode1': paranet_pi.loss_cce_mode1})
        net_pi_para2 = keras.models.load_model(tkp.net_path_pi_para2,
                                               custom_objects={'loss_cce_mode2': paranet_pi.loss_cce_mode2})
        net_pi_para3 = keras.models.load_model(tkp.net_path_pi_para3,
                                               custom_objects={'loss_cce_mode3': paranet_pi.loss_cce_mode3})
        net_pi_para4 = keras.models.load_model(tkp.net_path_pi_para4,
                                               custom_objects={'loss_cce_mode4': paranet_pi.loss_cce_mode4})
        net_pi_para5 = keras.models.load_model(tkp.net_path_pi_para5,
                                               custom_objects={'loss_cce_mode5': paranet_pi.loss_cce_mode5})
        one_step_net1, hidden_ex1 = one_step_model(net_pi_para1)
        one_step_net2, hidden_ex2 = one_step_model(net_pi_para2)
        one_step_net3, hidden_ex3 = one_step_model(net_pi_para3)
        one_step_net4, hidden_ex4 = one_step_model(net_pi_para4)
        one_step_net5, hidden_ex5 = one_step_model(net_pi_para5)
        nets.extend([one_step_net1, one_step_net2, one_step_net3, one_step_net4, one_step_net5])
        hidden_ex = [hidden_ex1, hidden_ex2, hidden_ex3, hidden_ex4, hidden_ex5]
    elif which_net == 'npi_int':
        net_npi_int = keras.models.load_model(tkp.net_path_npi_int)
        one_step_net, hidden_ex = one_step_model(net_npi_int)
        nets.append(one_step_net)
    elif which_net == 'npi_para':
        net_npi_para1 = keras.models.load_model(tkp.net_path_npi_para1,
                                                custom_objects={'loss_cce_mode1': paranet_npi.loss_cce_mode1})
        net_npi_para2 = keras.models.load_model(tkp.net_path_npi_para2,
                                                custom_objects={'loss_cce_mode2': paranet_npi.loss_cce_mode2})
        net_npi_para3 = keras.models.load_model(tkp.net_path_npi_para3,
                                                custom_objects={'loss_cce_mode3': paranet_npi.loss_cce_mode3})
        net_npi_para4 = keras.models.load_model(tkp.net_path_npi_para4,
                                                custom_objects={'loss_cce_mode4': paranet_npi.loss_cce_mode4})
        net_npi_para5 = keras.models.load_model(tkp.net_path_npi_para5,
                                                custom_objects={'loss_cce_mode5': paranet_npi.loss_cce_mode5})
        one_step_net1, hidden_ex1 = one_step_model(net_npi_para1)
        one_step_net2, hidden_ex2 = one_step_model(net_npi_para2)
        one_step_net3, hidden_ex3 = one_step_model(net_npi_para3)
        one_step_net4, hidden_ex4 = one_step_model(net_npi_para4)
        one_step_net5, hidden_ex5 = one_step_model(net_npi_para5)
        nets.extend([one_step_net1, one_step_net2, one_step_net3, one_step_net4, one_step_net5])
        hidden_ex = [hidden_ex1, hidden_ex2, hidden_ex3, hidden_ex4, hidden_ex5]
    else:
        raise ValueError('Error net structure')

    x0 = tkp.x0
    s0 = tkp.s0
    z0 = tkm.measurement_tracking(x0, 0)

    xtrue_all = np.zeros(shape=(run_batch, K + 1, tkp.nx))
    strue_all = np.zeros(shape=(run_batch, K + 1))
    xest_all = np.zeros(shape=(run_batch, K + 1, tkp.nx))
    xp_all = np.zeros(shape=(run_batch, K + 1, M, Np, tkp.nx))
    w_all = np.zeros(shape=(run_batch, K + 1, M, Np))
    mu_all = np.zeros(shape=(run_batch, K + 1, M))
    z_all = np.zeros(shape=(run_batch, K + 1, tkp.nz))
    cmtp_all = np.zeros(shape=(run_batch, K + 1, M, M, Np))
    what_all = np.zeros(shape=(run_batch, K + 1, M, M, Np))
    what_sum_all = np.zeros(shape=(run_batch, K + 1, M))
    zcliprd_all = np.zeros(shape=(run_batch, K + 1, M, M, Np))
    v_all = np.zeros(shape=(run_batch, K + 1, M, M, Np))
    xi_all = np.zeros(shape=(run_batch, K + 1, M, Np), dtype='int')
    zeta_all = np.zeros(shape=(run_batch, K + 1, M, Np), dtype='int')
    q_proposal_all = np.random.multivariate_normal(mean=np.zeros(tkp.nq), cov=tkp.Q, size=(K + 1, M, Np))

    data = np.load(tkp.data_path)
    x_data, z_data, s_data, t_data, tpm_data, ifreach_data, time_steps_data = tkplot.read_data(data)
    size_run = int(x_data.shape[0] * tkp.train_prop)
    size_run = int(x_data.shape[0] * tkp.train_prop) + 200
    size_run = 600
    size_run = tkp.size_run
    # xtrue_batch = np.swapaxes(x_data[size_run:, :, :], 1, 2)
    # ztrue_batch = np.swapaxes(z_data[size_run:, :, :], 1, 2)
    # strue_batch = s_data[size_run:, 0, :]
    # ttrue_batch = t_data[size_run:, 0, :]
    # ifreach_batch = ifreach_data[size_run:, 0, :]
    # time_steps_batch = time_steps_data[size_run:, 0, :]
    # time_steps = time_steps_batch[0, 0:K]

    xtrue_batch = np.swapaxes(x_data[tkp.sr, :, :], 1, 2)
    ztrue_batch = np.swapaxes(z_data[tkp.sr, :, :], 1, 2)
    strue_batch = s_data[tkp.sr, 0, :]
    ttrue_batch = t_data[tkp.sr, 0, :]
    time_steps_batch = time_steps_data[tkp.sr, 0, :]
    time_steps = time_steps_batch[0, 0:K]

    for n in tqdm(range(run_batch)):
        hidden_int_all = []
        hidden_para_all = [[]]

        xtrue_all[n, 0, :] = x0
        xtrue_all[n, 1:, :] = xtrue_batch[n, 0:K, :]
        z_all[n, 0, :] = z0
        z_all[n, 1:, :] = ztrue_batch[n, 0:K, :]
        strue_all[n, 0] = s0
        strue_all[n, 1:] = strue_batch[n, 0:K]

        # k=0:
        for j in range(M):
            for l in range(Np):
                w_all[n, 0, j, l] = 1 / Np
                xp_all[n, 0, j, l, :] = np.random.multivariate_normal(x0, tkp.Q0)
            mu_all[n, 0, j] = 0.96 if j == s0 - 1 else 0.01
        if which_net == 'pi_int' or which_net == 'npi_int':
            hidden0 = list()
            for l in range(len(hidden_ex)):
                hidden0.append(tf.convert_to_tensor(np.zeros(shape=hidden_ex[l].shape)))
            hidden_int_all.append(hidden0)
        elif which_net == 'pi_para' or which_net == 'npi_para':
            for j in range(M):
                hidden0 = list()
                for l in range(len(hidden_ex[j])):
                    hidden0.append(tf.convert_to_tensor(np.zeros(shape=hidden_ex[j][l].shape)))
                hidden_para_all[0].append(hidden0)

        for k in range(1, K + 1):
            zcli = np.zeros(shape=(M, Np))
            z_pre = z_all[n, k - 1, :]
            z = z_all[n, k, :]
            for i in range(M):
                xp_pre = xp_all[n, k - 1, i, :, :]
                if which_net == 'pi_int' or which_net == 'npi_int':
                    hidden_pre = hidden_int_all[k - 1]
                elif which_net == 'pi_para' or which_net == 'npi_para':
                    hidden_pre = hidden_para_all[k - 1][i]
                else:
                    raise ValueError('Error net structure')
                cmtp_pre, hidden_new = compute_cmtp(nets=nets, which_net=which_net,
                                                    x=xp_pre, z=z_pre, s=i,
                                                    hidden=hidden_pre)
                cmtp_all[n, k - 1, i, :, :] = cmtp_pre
                if which_net == 'pi_int' or which_net == 'npi_int':
                    if i == 0:
                        hidden_int_all.append(hidden_new)
                elif which_net == 'pi_para' or which_net == 'npi_para':
                    if i == 0:
                        hidden_para_all.append([])
                    hidden_para_all[k].append(hidden_new)
            # print(1)
            for j in range(M):
                for i in range(M):
                    for l in range(Np):
                        # what_pre = cmtp_all[n, k - 1, i, j, l] * mu_all[n, k - 1, i] * w_all[n, k - 1, i, l]
                        # what_all[n, k - 1, i, j, l] = what_pre
                        what_all[n, k - 1, i, j, l] = cmtp_all[n, k - 1, i, j, l] * mu_all[n, k - 1, i] * w_all[
                            n, k - 1, i, l]
                what_pre_sum = np.sum(what_all[n, k - 1, :, j, :])
                what_sum_all[n, k - 1, j] = what_pre_sum
                what_all[n, k - 1, :, j, :] = what_all[n, k - 1, :, j, :] / what_pre_sum
            # print(2)
            v_pre_sum = np.zeros(M)
            for j in range(M):
                for i in range(M):
                    for l in range(Np):
                        # zcliprd_pre = compute_zcpredict_likelihood(x_pre=xp_all[n, k - 1, i, l, :], z=z, s=j)
                        # zcliprd_all[n, k - 1, i, j, l] = zcliprd_pre
                        # v_pre = zcliprd_pre * what_all[n, k - 1, i, j, l]
                        # v_all[n, k - 1, i, j, l] = v_pre
                        v_all[n, k - 1, i, j, l] = compute_zcpredict_likelihood(x_pre=xp_all[n, k - 1, i, l, :], z=z,
                                                                                s=j) \
                                                   * what_all[n, k - 1, i, j, l]
                v_pre_sum[j] = np.sum(v_all[n, k - 1, :, j, :])
                v_all[n, k - 1, :, j, :] = v_all[n, k - 1, :, j, :] / v_pre_sum[j]
            # print(3)
            for j in range(M):
                xi_all[n, k - 1, j, :], zeta_all[n, k - 1, j, :] = sample_auxiliary_variables(v_all[n, k - 1, :, j, :])
                for l in range(Np):
                    xi = xi_all[n, k - 1, j, l]
                    zeta = zeta_all[n, k - 1, j, l]
                    xp = tkm.dynamic_tracking(sc=j + 1, x_p=xp_all[n, k - 1, xi, zeta, :],
                                              q=q_proposal_all[k - 1, j, l, :])
                    xp_all[n, k, j, l, :] = xp
                    zcli[j, l] = compute_zc_likelihood(x=xp, z=z, s=j)
                    w_all[n, k, j, l] = zcli[j, l] * what_all[n, k - 1, xi, j, zeta] / v_all[n, k - 1, xi, j, zeta]
                w_all[n, k, j, :] = w_all[n, k, j, :] / np.sum(w_all[n, k, j, :])
            # print(4)
            for j in range(M):
                li = np.sum(zcli[j, :]) / Np
                mu = li * what_sum_all[n, k - 1, j]
                # mu = v_pre_sum[j] * what_sum_all[n, k - 1, j]
                # for l2 in range(Np):
                #     mu = mu + w_all[n, k, j, l2] * what_sum_all[n, k - 1, j]
                mu_all[n, k, j] = mu
            mu_all[n, k, :] = mu_all[n, k, :] / np.sum(mu_all[n, k, :])
            # print(5)
            xest = np.zeros(tkp.nx)
            for j in range(M):
                xestj = np.zeros(tkp.nx)
                for l in range(Np):
                    xestj = xestj + w_all[n, k, j, l] * xp_all[n, k, j, l, :]
                xest = xest + mu_all[n, k, j] * xestj
            xest_all[n, k, :] = xest
            # print(6)

    np.savez(file=tkp.filter_data_path + '_' + which_net + '.npz',
             xtrue_all=xtrue_all,
             strue_all=strue_all,
             xest_all=xest_all,
             mu_all=mu_all,
             z_all=z_all,
             time_steps=time_steps)

    # ---------------------------------------------------------------------- #
    # -----------------------------|  E  N  D  |---------------------------- #
    # ---------------------------------------------------------------------- #



