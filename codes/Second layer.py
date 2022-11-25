"""
@File: First_layer.py
@Time: 2022/11/25
@Author:rufeng_lei@163.com
@desc:

"""

import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

from keras.layers import Input, Conv1D, AveragePooling1D, GlobalAveragePooling1D, \
    Dropout, Dense, Activation, Concatenate, Multiply, GlobalMaxPooling1D, Add, recurrent, \
    LSTM, Bidirectional, Conv2D, AveragePooling2D, BatchNormalization, Flatten, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Reshape, Permute, multiply, Lambda, add, subtract, MaxPooling2D, LeakyReLU, ELU
from keras.regularizers import l1, l2
from keras.optimizer_v2.adam import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from keras.models import Model, load_model
import tensorflow as tf
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import warnings

warnings.filterwarnings("ignore")


# Read DNA sequences
def read_fasta(fasta_file_name):
    seqs = []
    seqs_num = 0
    file = open(fasta_file_name)

    for line in file.readlines():
        if line.strip() == '':
            continue

        if line.startswith('>'):
            seqs_num = seqs_num + 1
            continue
        else:
            seq = line.strip()

            result1 = 'N' in seq
            result2 = 'n' in seq
            if result1 == False and result2 == False:
                seqs.append(seq)
    return seqs


# One-hot coding
def to_one_hot(seqs):
    base_dict = {
        'a': 0, 'c': 1, 'g': 2, 't': 3,
        'A': 0, 'C': 1, 'G': 2, 'T': 3
    }

    one_hot_4_seqs = []
    for seq in seqs:

        one_hot_matrix = np.zeros([4, len(seq)], dtype=float)
        index = 0
        for seq_base in seq:
            one_hot_matrix[base_dict[seq_base], index] = 1
            index = index + 1

        one_hot_4_seqs.append(one_hot_matrix)
    return one_hot_4_seqs


# NCP coding
def to_properties_code(seqs):
    properties_code_dict = {
        'A': [1, 1, 1], 'C': [0, 1, 0], 'G': [1, 0, 0], 'T': [0, 0, 1],
        'a': [1, 1, 1], 'c': [0, 1, 0], 'g': [1, 0, 0], 't': [0, 0, 1]
    }
    properties_code = []
    for seq in seqs:
        properties_matrix = np.zeros([3, len(seq)], dtype=float)
        m = 0
        for seq_base in seq:
            properties_matrix[:, m] = properties_code_dict[seq_base]
            m = m + 1
        properties_code.append(properties_matrix)
    return properties_code

# Performance evaluation
def show_performance(y_true, y_pred):
    # 定义tp, fp, tn, fn初始值
    TP, FP, FN, TN = 0, 0, 0, 0

    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] > 0.5:
                TP += 1
            else:
                FN += 1
        if y_true[i] == 0:
            if y_pred[i] > 0.5:
                FP += 1
            else:
                TN += 1

    # 计算敏感性Sn
    Sn = TP / (TP + FN + 1e-06)
    # 计算特异性Sp
    Sp = TN / (FP + TN + 1e-06)
    # 计算Acc值
    Acc = (TP + TN) / len(y_true)
    # 计算MCC：马修斯相关系数是在混淆矩阵环境中建立二元分类器预测质量的最具信息性的单一分数
    MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-06)

    return Sn, Sp, Acc, MCC


# Define a single convolutional layer in a dense convolutional block
def conv_factory(x, filters, dropout_rate, weight_decay=1e-4):
    x = Activation('relu')(x)
    x = Conv2D(filters=filters,
               kernel_size=(3, 3),
               kernel_initializer="he_normal",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(dropout_rate)(x)
    return x


# Defining the transition layer
def transition(x, filters, dropout_rate, weight_decay=1e-4):
    # x = Activation('relu')(x)
    x = Conv2D(filters=filters,
               kernel_size=(1, 1),
               kernel_initializer="he_normal",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(dropout_rate)(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = BatchNormalization(axis=-1)(x)
    return x


# Define dense convolution blocks
def denseblock(x, layers, filters, growth_rate, dropout_rate=None, weight_decay=1e-4):
    list_feature_map = [x]
    # 循环三次conv_factory部分
    for i in range(layers):
        x = conv_factory(x, growth_rate,
                         dropout_rate, weight_decay)
        list_feature_map.append(x)
        x = Concatenate(axis=-1)(list_feature_map)
        filters = filters + growth_rate
    return x, filters

# Building the model
def build_model(windows=7, denseblocks=4, layers=3, filters=96,
                growth_rate=32, dropout_rate=0.2, weight_decay=1e-4):
    input_1 = Input(shape=(windows, 200, 1))  # Input的本质是实例化一个Keras Tensor，就是你是输入一条序列的shape大小

    for i in range(denseblocks - 1):
        # Add denseblock
        x_1, filters_1 = denseblock(input_1, layers=layers,
                                    filters=filters, growth_rate=growth_rate,
                                    dropout_rate=dropout_rate, weight_decay=weight_decay)
        # Add BatchNormalization
        x_1 = BatchNormalization(axis=-1)(x_1)

        # Add transition
        x_1 = transition(x_1, filters=filters_1,
                         dropout_rate=dropout_rate, weight_decay=weight_decay)
    # The last denseblock
    # Add denseblock
    x_1, filters_1 = denseblock(x_1, layers=layers,
                                filters=filters, growth_rate=growth_rate,
                                dropout_rate=dropout_rate, weight_decay=weight_decay)
    # Add BatchNormalization
    x_1 = BatchNormalization(axis=-1)(x_1)

    # Pooling
    x_1 = AveragePooling2D(pool_size=(1, 12), strides=(1, 1))(x_1)

    # Flatten
    x = Flatten()(x_1)

    # MLP
    x = Dense(units=240, activation="sigmoid", use_bias=False,
              kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay))(x)

    x = Dropout(0.5)(x)

    x = Dense(units=40, activation="sigmoid", use_bias=False,
              kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay))(x)

    x = Dropout(0.2)(x)

    x = Dense(units=2, activation="softmax", use_bias=False,
              kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay))(x)

    inputs = [input_1]
    outputs = [x]

    model = Model(inputs=inputs, outputs=outputs, name="enhancer")

    optimizer = Adam(lr=1e-4, epsilon=1e-8)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def performance_mean(performance):
    print('Sn = %.4f ± %.4f' % (np.mean(performance[:, 0]), np.std(performance[:, 0])))
    print('Sp = %.4f ± %.4f' % (np.mean(performance[:, 1]), np.std(performance[:, 1])))
    print('Acc = %.4f ± %.4f' % (np.mean(performance[:, 2]), np.std(performance[:, 2])))
    print('Mcc = %.4f ± %.4f' % (np.mean(performance[:, 3]), np.std(performance[:, 3])))
    print('Auc = %.4f ± %.4f' % (np.mean(performance[:, 4]), np.std(performance[:, 4])))


if __name__ == '__main__':

    np.random.seed(0)
    tf.random.set_seed(1)  # for reproducibility

    # Read the training set
    train_strong_seqs = read_fasta('F:/py-file/leirufeng/two_enhancer/data/layer2/train_strong_enhancers.fa')
    train_weak_seqs = read_fasta('F:/py-file/leirufeng/two_enhancer/data/layer2/train_weak_enhancers.fa')

    train_seqs = np.concatenate((train_strong_seqs, train_weak_seqs), axis=0)

    train_onehot = np.array(to_one_hot(train_seqs)).astype(np.float32)
    train_properties_code = np.array(to_properties_code(train_seqs)).astype(np.float32)

    train = np.concatenate((train_onehot, train_properties_code), axis=1)

    train_label = np.array([1] * 742 + [0] * 742).astype(np.float32)
    train_label = to_categorical(train_label, num_classes=2)

    # Read the testing set
    test_strong_seqs = read_fasta('F:/py-file/leirufeng/two_enhancer/data/layer2/test_strong_enhancers.fa')
    test_weak_seqs = read_fasta('F:/py-file/leirufeng/two_enhancer/data/layer2/test_weak_enhancers.fa')

    test_seqs = np.concatenate((test_strong_seqs, test_weak_seqs), axis=0)

    test_onehot = np.array(to_one_hot(test_seqs)).astype(np.float32)
    test_properties_code = np.array(to_properties_code(test_seqs)).astype(np.float32)

    test = np.concatenate((test_onehot, test_properties_code), axis=1)

    test_label = np.array([1] * 100 + [0] * 100).astype(np.float32)
    test_label = to_categorical(test_label, num_classes=2)

    # Cross-validation
    n = 5
    k_fold = KFold(n_splits=n, shuffle=True, random_state=42)

    sv_10_result = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    print(np.array(sv_10_result).shape)

    for k in range(10):
        print('*' * 30 + ' the ' + str(k) + ' cycle ' + '*' * 30)
        all_Sn = []
        all_Sp = []
        all_Acc = []
        all_MCC = []
        all_AUC = []
        test_pred_all = []
        mean_fpr = np.linspace(0, 1, 100)
        for fold_count, (train_index, val_index) in enumerate(k_fold.split(train)):
            print('*' * 30 + ' fold ' + str(fold_count) + ' ' + '*' * 30)
            trains, val = train[train_index], train[val_index]
            trains_label, val_label = train_label[train_index], train_label[val_index]

            model = build_model()

            BATCH_SIZE = 30
            EPOCHS = 300

            history = model.fit(x=trains, y=trains_label, validation_data=(val, val_label), epochs=EPOCHS,
                                batch_size=BATCH_SIZE, shuffle=True,
                                callbacks=[EarlyStopping(monitor='val_loss', patience=40, mode='auto')],
                                verbose=1)  # callbacks回调，将数据传给history

            with open('F:/py-file/leirufeng/two_enhancer/files/log_history/one_log_history.txt', 'w') as f:
                f.write(str(history.history))

            train_loss = history.history["loss"]
            train_acc = history.history["accuracy"]
            val_loss = history.history["val_loss"]
            val_acc = history.history["val_accuracy"]

            loss, accuracy = model.evaluate(val, val_label, verbose=1)

            print('val loss:', loss)
            print('val accuracy:', accuracy)

            model.save('F:/py-file/leirufeng/two_enhancer/models/one_enhancer_model_' + str(fold_count) + '.h5')

            del model

            model = load_model('F:/py-file/leirufeng/two_enhancer/models/one_enhancer_model_' + str(fold_count) + '.h5')

            test_pred = model.predict(test, verbose=1)
            test_pred_all.append(test_pred[:, 1])

            # Sn, Sp, Acc, MCC, AUC
            Sn, Sp, Acc, MCC = show_performance(test_label[:, 1], test_pred[:, 1])
            AUC = roc_auc_score(test_label[:, 1], test_pred[:, 1])
            print('Sn = %f, Sp = %f, Acc = %f, MCC = %f, AUC = %f' % (Sn, Sp, Acc, MCC, AUC))

            # Put each collapsed evaluation metric into a master list

            all_Sn.append(Sn)
            all_Sp.append(Sp)
            all_Acc.append(Acc)
            all_MCC.append(MCC)
            all_AUC.append(AUC)
            fold_count += 1
        fold_avg_Sn = np.mean(all_Sn)
        fold_avg_Sp = np.mean(all_Sp)
        fold_avg_Acc = np.mean(all_Acc)
        fold_avg_MCC = np.mean(all_MCC)
        fold_avg_AUC = np.mean(all_AUC)

        # soft voting
        test_pred_all = np.array(test_pred_all).T

        ruan_voting_test_pred = test_pred_all.mean(axis=1)

        sv_Sn, sv_Sp, sv_Acc, sv_MCC = show_performance(test_label[:, 1], ruan_voting_test_pred)
        sv_AUC = roc_auc_score(test_label[:, 1], ruan_voting_test_pred)
        sv_result = [sv_Sn, sv_Sp, sv_Acc, sv_MCC, sv_AUC]
        sv_10_result.append(sv_result)

        '''Mapping the ROC'''
        fpr, tpr, thresholds = roc_curve(test_label[:, 1], ruan_voting_test_pred, pos_label=1)

        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        plt.plot(fpr, tpr, label='ROC cycle {} (AUC={:.4f})'.format(str(k), sv_AUC))

    print('---------------------------------------------soft voting 10---------------------------------------')
    print(np.array(sv_10_result))
    performance_mean(np.array(sv_10_result))

    '''Mapping the ROC'''
    plt.plot([0, 1], [0, 1], '--', color='red')
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    mean_auc = np.mean(np.array(sv_10_result)[:, 4])


    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC=%0.4f)' % (mean_auc), lw=2, alpha=.8)

    plt.title('ROC Curve of Second Layer')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend(loc='lower right')
    plt.savefig('F:/py-file/leirufeng/two_enhancer/images/ROC_Curve_of_Second_Layer.jpg', dpi=1200, bbox_inches='tight')
    plt.show()


