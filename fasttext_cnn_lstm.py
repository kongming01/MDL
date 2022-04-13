import numpy as np
from keras.layers import Embedding, MaxPooling1D, Conv1D, LSTM, Flatten
from keras.layers import Dropout, Input, Dense, Add, Concatenate, GlobalAveragePooling1D
from keras.models import Model
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

import argparse
import time, os
# import random
from datetime import datetime

from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, \
    f1_score, precision_score, recall_score, average_precision_score, precision_recall_curve, auc

# from keras.utils.np_utils import to_categorical
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import GridSearchCV, train_test_split

import data_helpers


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Training model
# ==================================================
def prepare_inputs(training_ratio, data, idx_run, log):
    print("preparing inputs")
    print('used seconds: ', time.time() - start_time)

    log.writelines("\n" + "preparing inputs ")

    x, y, vocabulary_length, embedding_weights = data

    x_text, x_link = x

    print('x_text.shape: ', x_text.shape)
    print('x_link.shape: ', x_link.shape)
    print('y.shape: ', y.shape)

    sequence_length = (x_text.shape[1], x_link.shape[1])
    class_num = y.shape[1]

    # Shuffle data
    np.random.seed(idx_run + nb_seed)  # the num should be same as that in other baseline, added by Zhang
    shuffle_indices = np.random.permutation(np.arange(len(y)))

    log.writelines('\n' + 'shuffle_indices[:10] ' + str(shuffle_indices[:10]))

    x_text_shuffled = x_text[shuffle_indices]
    x_link_shuffled = x_link[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # training_ratio = 0.5
    num_training = int(np.ceil(x_text_shuffled.shape[0] * training_ratio))

    x_text_train = x_text_shuffled[:num_training]
    x_link_train = x_link_shuffled[:num_training]
    y_train = y_shuffled[:num_training]

    print("num_training: ", num_training)
    log.writelines("\n" + "num_training: " + str(num_training))

    x_text_test = x_text_shuffled[num_training:]
    x_link_test = x_link_shuffled[num_training:]
    y_test = y_shuffled[num_training:]

    x_list = (x_text_train, x_link_train, x_text_test, x_link_test)
    y_list = (y_train, y_test)
    text_statistics = (vocabulary_length, sequence_length, class_num)

    return x_list, y_list, embedding_weights, text_statistics


def product(y_true, y_pred):
    return K.mean((1 - y_pred * y_true), axis=-1)


def make_model(sequence_length, vocabulary_length, class_num, embedding_dim,
               output_dim_lstm, dropout_prob_0, dropout_prob_1, hidden_dims,
               num_filters, filter_sizes, embedding_weights):
    # seq_length = 10
    input_a = Input(shape=(sequence_length[0],))

    if embedding_dim > embedding_weights.shape[1]:
        embedding_dim = embedding_weights.shape[1]

    emb1 = Embedding(vocabulary_length[0] + 1, embedding_dim, init='he_normal', input_length=sequence_length[0],
                     weights=[embedding_weights[:, :embedding_dim]], dropout=dropout_prob_0)(input_a)

    out1 = Conv1D(nb_filter=num_filters, filter_length=filter_sizes,
                 border_mode='valid', activation='relu')(emb1)
    out1 = MaxPooling1D(pool_size=sequence_length[0] - filter_sizes + 1)(out1)

    # out1 = LSTM(output_dim_lstm, activation='tanh',
    #             dropout_W=dropout_prob_1, dropout_U=dropout_prob_1)(out1)

    out_a = Flatten()(out1)
    # out_a = Dense(hidden_dims, init='he_normal', activation='tanh')(out1)


    # input_b = Input(shape=(sequence_length[1],))
    # out2 = Embedding(vocabulary_length[1] + 1, embedding_dim, init='he_normal', input_length=sequence_length[1],
    #                  weights=None, dropout=dropout_prob_0)(input_b)

    # out2 = Conv1D(nb_filter=num_filters, filter_length=filter_sizes,
    #               border_mode='valid', activation='relu')(out2)
    # out2 = MaxPooling1D(pool_length=class_num)(out2)

    out2 = LSTM(output_dim_lstm, activation='tanh',
                dropout_W=dropout_prob_1, dropout_U=dropout_prob_1, return_sequences=True)(emb1)

    # out2 = LSTM(embedding_dim, activation='tanh',
    #             dropout_W=dropout_prob_1, dropout_U=dropout_prob_1, return_sequences=True)(out2)

    # out_b = Dense(hidden_dims, init='he_normal', activation='tanh')(out2)
    out_b = Flatten()(out2)


    out_c = GlobalAveragePooling1D()(emb1)

    # merged = Add()([out_a, out_b]) # for Keras 2.2

    # merged = merge([out_a, out_b], mode='sum')
    # merged = Add()([out_a, out_b])
    merged = Concatenate()([out_a, out_b, out_c])

    # merged = merge([out_a, out_b], mode='concat', concat_axis=-1) #  worse than 'sum'

    main_output = Dense(class_num, activation='softmax', name='output')(merged)  # define a new activation
    main_model = Model(input=input_a, output=main_output)

    # adam = Adam(lr=0.0005, beta_1=0.7, beta_2=0.9, epsilon=1e-08, decay=0.0)
    # main_model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['acc'])

    main_model.compile(optimizer='adagrad', loss=['categorical_crossentropy'], metrics=['acc'])

    # print(main_model.summary())

    return main_model


def is_member(array_a, array_b):
    c = np.array([np.sum(i == array_b) for i in array_a])
    c = (c == 1)
    return c


def get_evaluation(pred, y_test, log):
    y_pred = []
    y_testing = []
    for i in range(len(pred)):
        y_pred.append(pred[i].argmax())
        y_testing.append(y_test[i].argmax())

    accuracy = accuracy_score(y_testing, y_pred)

    pre, rec, thresholds = precision_recall_curve(y_testing, y_pred)
    pr_auc = auc(pre, rec)

    roc_auc = roc_auc_score(y_testing, y_pred, average='macro')

    binary_precision = precision_score(y_testing, y_pred, average='binary')
    binary_recall = recall_score(y_testing, y_pred, average='binary')
    binary_f1 = f1_score(y_testing, y_pred, average='binary')

    precision = precision_score(y_testing, y_pred, average='macro')
    recall = recall_score(y_testing, y_pred, average='macro')
    f1 = f1_score(y_testing, y_pred, average='macro')

    average_precision = average_precision_score(y_testing, y_pred, average='macro')

    weighted_f1 = f1_score(y_testing, y_pred, average='weighted')

    # print('test data length: ', len(y_testing))
    # print('accuracy: ', accuracy)
    # print('average_precision: ', average_precision)
    # print('f1: ', f1)
    # print('precision: ', precision)
    # print('recall: ', recall)
    # print(classification_report(y_testing, y_pred))
    #
    # print('weighted_f1: ', weighted_f1)
    # print('pr_auc: ', pr_auc)
    # print('roc_auc: ', roc_auc)

    log.writelines("\n" + 'test data length: ' + str(len(y_testing)))

    log.writelines("\n" + 'accuracy: ' + str(accuracy))
    log.writelines("\n" + 'pr_auc: ' + str(pr_auc))
    log.writelines("\n" + 'roc_auc: ' + str(roc_auc))

    log.writelines("\n" + 'binary_precision: ' + str(binary_precision))
    log.writelines("\n" + 'binary_recall: ' + str(binary_recall))
    log.writelines("\n" + 'binary_f1: ' + str(binary_f1))

    log.writelines("\n" + 'precision: ' + str(precision))
    log.writelines("\n" + 'recall: ' + str(recall))
    log.writelines("\n" + 'f1: ' + str(f1))

    log.writelines("\n" + 'average_precision: ' + str(average_precision))

    log.writelines("\n" + 'weighted_f1: ' + str(weighted_f1))

    log.writelines("\n" + classification_report(y_testing, y_pred))

    log.flush()

    return accuracy, pr_auc, roc_auc, binary_precision, binary_recall, binary_f1, precision, recall, f1, \
           average_precision, weighted_f1


# Experimental parameters
parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, help="name of a dataset")
parser.add_argument("num_epochs", type=int, help="number of epochs")
parser.add_argument("training_validation_ratio", type=float, help="ratio of training and validation data")
parser.add_argument("splitting_ratio", type=float, help="ratio of splitting training and validation data")
parser.add_argument("result_file", type=str, help="result file")

args = parser.parse_args()

dataset = args.dataset

training_validation_ratio = args.training_validation_ratio
training_ratio = [training_validation_ratio]  # [0.1, 0.2, 0.3, 0.4, 0.5]
val_split = args.splitting_ratio

# training_ratio = [0.75]  # [0.1, 0.2, 0.3, 0.4, 0.5]
nb_run = 21
nb_rand_search = 100
nb_seed = 2017

# Training parameters
batch_size = 128  # little impact
num_epochs = args.num_epochs

# val_split = 0.67

parameter_set = list(range(32, 1056, 32))
# parameter_set = [32, 64, 128, 256, 512, 1024]

dropout_prob_embedding = np.arange(0.2, 0.55, 0.05)
dropout_prob_lstm = np.arange(0.2, 0.55, 0.05)

filter_sizes_list = [3, 4, 5]
num_filters_list = [32, 64, 128, 256, 512]

filter_sizes_best = filter_sizes_list[0]
num_filters_best = num_filters_list[0]

high_val = 0
embedding_dim_best = parameter_set[0]
output_dim_lstm_best = parameter_set[0]
hidden_dims_best = parameter_set[0]
num_epochs_best = num_epochs
dropout_prob_emb_best = dropout_prob_embedding[0]
dropout_prob_lstm_best = dropout_prob_lstm[0]

result_file = args.result_file + ".txt"
f_out = open(result_file, 'a')

# Load data
start_time = time.time()

print('start time: ', datetime.now())
f_out.writelines("\n\nMulti-view-clstm-v8-w2v \n" + 'start time: ' + str(datetime.now()))

print("Loading data ...")
f_out.writelines("\n" + "Loading data ...")

data = data_helpers.load_data11(dataset, f_out)

# earlyStopping_cv = EarlyStopping(monitor='val_acc', patience=5, verbose=2, mode='auto')
# earlyStopping_test = EarlyStopping(monitor='acc', patience=5, verbose=2, mode='auto')

for r in training_ratio:
    acc_list = []
    auc_list = []
    roc_auc_list = []

    binary_precision_list = []
    binary_recall_list = []
    binary_f1_list = []

    precision_list = []
    recall_list = []
    f1_list = []

    weighted_f1_list = []
    average_precision_list = []

    for i in range(nb_run):
        if i != 0:
            f_out.writelines("\n\n" + "Trial " + str(i))

        (x_list, y_list, embedding_weights, text_statistics) = prepare_inputs(r, data, i, f_out)
        (x_text_train, x_link_train, x_text_test, x_link_test) = x_list
        (y_train, y_test) = y_list
        (vocabulary_length, sequence_length, class_num) = text_statistics

        num_data_train = len(y_train)
        print('num_data_train: ', num_data_train)

        print("\n" + "Preparing input data is over!")
        print("used time: ", time.time() - start_time)

        f_out.writelines("\n" + "training_validation_ratio: " + str(training_validation_ratio))
        f_out.writelines("\n" + "val_split: " + str(val_split))
        f_out.writelines("\n" + "num_epochs: " + str(num_epochs))
        f_out.writelines("\n" + "nb_rand_search: " + str(nb_rand_search))
        f_out.writelines("\n" + "nb_seed: " + str(nb_seed))

        f_out.writelines("\n" + "Preparing input data is over!")
        f_out.writelines("\n" + "used time: " + str(time.time() - start_time))

        # get the indexes of training data for cross-validation
        np.random.seed(i + nb_seed)
        tmp = np.random.choice(np.arange(1, num_data_train + 1), int(np.ceil(val_split * num_data_train)),
                               replace=False)

        f_out.writelines('\n' + 'tmp[:10] ' + str(tmp[:10]))

        idx_train = is_member(np.arange(1, num_data_train + 1), tmp)

        x_text_train_tr = x_text_train[idx_train]
        x_link_train_tr = x_link_train[idx_train]
        y_train_tr = y_train[idx_train]

        print('x_text_train_tr.shape: ', x_text_train_tr.shape)
        print('x_link_train_tr.shape: ', x_link_train_tr.shape)
        print('y_train_tr.shape: ', y_train_tr.shape)

        x_text_train_val = x_text_train[~idx_train]
        x_link_train_val = x_link_train[~idx_train]
        y_train_val = y_train[~idx_train]

        val_data = x_text_train_val, y_train_val

        print('len(y_train_tr): ', len(y_train_tr))
        print('len(y_train_val): ', len(y_train_val))

        f_out.writelines("\n" + 'len(y_train_tr): ' + str(len(y_train_tr)))
        f_out.writelines("\n" + 'len(y_train_val): ' + str(len(y_train_val)))

        print("Trial ", i)
        print("starting cross-validation: ")
        print("used time: ", time.time() - start_time)

        f_out.writelines("\n" + "starting cross-validation: ")
        f_out.writelines("\n" + "used time: " + str(time.time() - start_time))

        f_out.flush()

        if i == 0:
            best_acc, best_pr_auc, best_roc_auc = (0, 0, 0)
            best_binary_pre, best_binary_recall, best_binary_f1 = (0, 0, 0)
            best_precision, best_recall, best_mac_f1, best_average_pre, best_weighted_f1 = (0, 0, 0, 0, 0)

            best_acc_val = 0
            for j in range(nb_rand_search):
                e_dim = int(np.random.choice(parameter_set, 1, replace=False))
                if e_dim == 32:
                    tmp_set = [32]
                else:
                    tmp_set = [32] + list(range(64, e_dim + 64, 64))
                o_dim = int(np.random.choice(tmp_set, 1, replace=False))

                if o_dim == 32:
                    tmp_set = [32]
                else:
                    tmp_set = [32] + list(range(64, o_dim + 64, 64))

                h_dim = int(np.random.choice(tmp_set, 1, replace=False))

                p_embedding = float(np.random.choice(dropout_prob_embedding, 1, replace=False))
                p_lstm = float(np.random.choice(dropout_prob_lstm, 1, replace=False))

                num_filters_trial = int(np.random.choice(num_filters_list, 1, replace=False))
                filter_sizes_trial = int(np.random.choice(filter_sizes_list, 1, replace=False))

                f_out.writelines("\n\n" + 'tuning parameters, ' + 'round ' + str(j))
                f_out.writelines("\n" + 'embedding_dim: ' + str(e_dim))
                f_out.writelines("\n" + 'output_dim_lstm: ' + str(o_dim))
                f_out.writelines("\n" + 'hidden_dims: ' + str(h_dim))
                f_out.writelines("\n" + 'dropout_prob_embedding: ' + str(p_embedding))
                f_out.writelines("\n" + 'dropout_prob_lstm: ' + str(p_lstm))
                f_out.writelines("\n" + 'num_filters_trial: ' + str(num_filters_trial))
                f_out.writelines("\n" + 'filter_sizes_trial: ' + str(filter_sizes_trial))

                checkpointer = ModelCheckpoint(filepath=result_file + ".hdf5", monitor='val_acc', verbose=1,
                                               save_best_only=True, save_weights_only=True)

                tmp_model = make_model(sequence_length, vocabulary_length, class_num, e_dim, o_dim,
                                    p_embedding, p_lstm, h_dim, num_filters_trial, filter_sizes_trial, embedding_weights)

                hist = tmp_model.fit(x_text_train_tr, y_train_tr, batch_size=batch_size,
                                     nb_epoch=num_epochs, verbose=2, validation_data=val_data,
                                     callbacks=[checkpointer])

                val_acc = np.array(hist.history['val_acc'])

                high_acc = max(val_acc)
                high_epoch = val_acc.argmax() + 1

                print('epoch: ', high_epoch)
                print('max accuracy: ', high_acc)

                f_out.writelines("\n" + 'best_epoch: ' + str(high_epoch))
                f_out.writelines("\n" + 'best_acc: ' + str(high_acc))

                f_out.flush()

                tmp_model.load_weights(result_file + ".hdf5")

                pred = tmp_model.predict(x_text_train_val)

                metrics = get_evaluation(pred, y_train_val, f_out)
                acc, pr_auc, roc_auc, bin_pre, bin_recall, bin_f1, pre, recall, f1, ave_pre, w_f1 = metrics

                os.remove(result_file + ".hdf5")

                if high_acc > best_acc_val:
                    best_acc_val = high_acc

                    best_acc = acc
                    best_pr_auc = pr_auc
                    best_roc_auc = roc_auc

                    best_binary_pre = bin_pre
                    best_binary_recall = bin_recall
                    best_binary_f1 = bin_f1

                    best_precision = pre
                    best_recall = recall
                    best_mac_f1 = f1

                    best_average_pre = ave_pre
                    best_weighted_f1 = w_f1

                    num_epochs_best = high_epoch
                    embedding_dim_best = e_dim
                    output_dim_lstm_best = o_dim
                    hidden_dims_best = h_dim
                    dropout_prob_emb_best = p_embedding
                    dropout_prob_lstm_best = p_lstm
                    num_filters_best = num_filters_trial
                    filter_sizes_best = filter_sizes_trial

            # acc_list.append(best_acc)
            # f1_list.append(best_average_pre)
            # f1_list.append(best_mac_f1)
            # precision_list.append(best_precision)
            # recall_list.append(best_recall)
            #
            # weighted_f1_list.append(best_weighted_f1)
            # auc_list.append(best_auc)
            # roc_auc_list.append(best_roc_auc)

            f_out.flush()
            continue

        print('batch_size: ', batch_size)
        print('val_split: ', val_split)

        print('embedding_dim_best: ', embedding_dim_best)
        print('output_dim_lstm_best: ', output_dim_lstm_best)
        print('hidden_dims_best: ', hidden_dims_best)

        f_out.writelines("\n" + "used time: " + str(time.time() - start_time))

        f_out.writelines("\n\n" + 'num_epochs: ' + str(num_epochs))
        f_out.writelines("\n" + 'nb_run: ' + str(nb_run))
        f_out.writelines("\n" + 'batch_size: ' + str(batch_size))
        f_out.writelines("\n" + 'val_split: ' + str(val_split))

        f_out.writelines("\n" + 'embedding_dim_best: ' + str(embedding_dim_best))
        f_out.writelines("\n" + 'output_dim_lstm_best: ' + str(output_dim_lstm_best))
        f_out.writelines("\n" + 'hidden_dims_best: ' + str(hidden_dims_best))
        f_out.writelines("\n" + 'dropout_prob_emb_best: ' + str(dropout_prob_emb_best))
        f_out.writelines("\n" + 'dropout_prob_lstm: ' + str(dropout_prob_lstm_best))
        f_out.writelines("\n" + 'num_filters_best: ' + str(num_filters_best))
        f_out.writelines("\n" + 'filter_sizes_best: ' + str(filter_sizes_best))

        print("building and fitting model: ")
        print("used time: ", time.time() - start_time)

        f_out.writelines('\n\n building and fitting model: ')

        best_model = make_model(sequence_length, vocabulary_length, class_num, embedding_dim_best,
                                output_dim_lstm_best, dropout_prob_emb_best, dropout_prob_lstm_best, hidden_dims_best,
                                num_filters_best, filter_sizes_best, embedding_weights)

        hist = best_model.fit(x_text_train_tr, y_train_tr, batch_size=batch_size,
                              nb_epoch=num_epochs_best, verbose=2, validation_data=val_data)

        val_acc = np.array(hist.history['val_acc'])

        print('epoch: ', val_acc.argmax() + 1)
        print('max accuary: ', max(val_acc))

        print("evaluating and predicting: ")
        print("used time: ", time.time() - start_time)
        f_out.writelines("\n" + "evaluating and predicting: ")
        f_out.writelines("\n" + "used time: " + str(time.time() - start_time))

        acc1 = best_model.evaluate(x_text_test, y_test, batch_size=batch_size)

        print('len(y_test0): ', len(y_test))
        print('\n sum(y_test): ', sum(y_test))
        print('Test accuracy: ', acc1)

        f_out.writelines("\n" + 'Test accuracy: ' + str(acc1))

        pred = best_model.predict(x_text_test)

        print(len(pred), len(pred[0]))

        acc, pr_auc, roc_auc, bin_pre, bin_recall, bin_f1, pre, recall, f1, ave_pre, w_f1 = get_evaluation(pred, y_test,
                                                                                                           f_out)

        acc_list.append(acc)
        auc_list.append(pr_auc)
        roc_auc_list.append(roc_auc)

        binary_precision_list.append(bin_pre)
        binary_recall_list.append(bin_recall)
        binary_f1_list.append(bin_f1)

        precision_list.append(pre)
        recall_list.append(recall)
        f1_list.append(f1)

        average_precision_list.append(ave_pre)
        weighted_f1_list.append(w_f1)

        f_out.flush()

    f_out.writelines("\n\n" + str(nb_run) + " trials are over! ")
    f_out.writelines("\n" + str(time.time() - start_time) + " seconds are used!")

    f_out.writelines("\n\n" + 'training ratio: ' + str(r) + '  accuracy: ' + str(np.round(acc_list, 3)))
    f_out.writelines("\n" + 'training ratio: ' + str(r) + '  auc: ' + str(np.round(auc_list, 3)))
    f_out.writelines("\n" + 'training ratio: ' + str(r) + '  roc_auc: ' + str(np.round(roc_auc_list, 3)))

    f_out.writelines(
        "\n" + 'training ratio: ' + str(r) + '  binary_precision: ' + str(np.round(binary_precision_list, 3)))
    f_out.writelines("\n" + 'training ratio: ' + str(r) + '  binary_recall: ' + str(np.round(binary_recall_list, 3)))
    f_out.writelines("\n" + 'training ratio: ' + str(r) + '  binary_f1: ' + str(np.round(binary_f1_list, 3)))

    f_out.writelines("\n" + 'training ratio: ' + str(r) + '  mac_precision: ' + str(np.round(precision_list, 3)))
    f_out.writelines("\n" + 'training ratio: ' + str(r) + '  mac_recall: ' + str(np.round(recall_list, 3)))
    f_out.writelines("\n" + 'training ratio: ' + str(r) + '  f1: ' + str(np.round(f1_list, 3)))

    f_out.writelines(
        "\n" + 'training ratio: ' + str(r) + '  average_precision: ' + str(np.round(average_precision_list, 3)))
    f_out.writelines("\n" + 'training ratio: ' + str(r) + '  weighted_f1: ' + str(np.round(weighted_f1_list, 3)))

    f_out.writelines("\n\n" + 'training ratio: ' + str(r) + '  mean accuracy: ' + str(np.round(np.mean(acc_list), 3)))
    f_out.writelines("\n" + 'training ratio: ' + str(r) + '  mean auc: ' + str(np.round(np.mean(auc_list), 3)))
    f_out.writelines(
        "\n" + 'training ratio: ' + str(r) + '  mean roc_auc: ' + str(np.round(np.mean(roc_auc_list), 3)))

    f_out.writelines(
        "\n" + 'training ratio: ' + str(r) + '  mean binary_precision: ' + str(
            np.round(np.mean(binary_precision_list), 3)))
    f_out.writelines(
        "\n" + 'training ratio: ' + str(r) + '  mean binary_recall: ' + str(np.round(np.mean(binary_recall_list), 3)))
    f_out.writelines(
        "\n" + 'training ratio: ' + str(r) + '  mean binary_f1: ' + str(np.round(np.mean(binary_f1_list), 3)))

    f_out.writelines(
        "\n" + 'training ratio: ' + str(r) + '  mean mac_precision: ' + str(np.round(np.mean(precision_list), 3)))
    f_out.writelines(
        "\n" + 'training ratio: ' + str(r) + '  mean mac_recall: ' + str(np.round(np.mean(recall_list), 3)))
    f_out.writelines(
        "\n" + 'training ratio: ' + str(r) + '  mean f1: ' + str(np.round(np.mean(f1_list), 3)))

    f_out.writelines(
        "\n" + 'training ratio: ' + str(r) + '  mean average_precision: ' + str(
            np.round(np.mean(average_precision_list), 3)))
    f_out.writelines(
        "\n" + 'training ratio: ' + str(r) + '  mean weighted_f1: ' + str(np.round(np.mean(weighted_f1_list), 3)))

    f_out.flush()

print('end time: ', datetime.now())
f_out.writelines("\n\n" + 'end time: ' + str(datetime.now()))

f_out.flush()
f_out.close()
