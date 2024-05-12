"""
@File: First_layer.py
@Time: 2022/11/25
@Author:rufeng_lei@163.com
@desc:

"""
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
    train_pos_seqs = np.array(read_fasta('../data/layer1/train_enhancers.fa'))
    train_neg_seqs = np.array(read_fasta('../data/layer1/train_nonenhancers.fa'))

    train_seqs = np.concatenate((train_pos_seqs, train_neg_seqs), axis=0)


    train_onehot = np.array(to_one_hot(train_seqs)).astype(np.float32)
    train_properties_code = np.array(to_properties_code(train_seqs)).astype(np.float32)

    train = np.concatenate((train_onehot, train_properties_code), axis=1)

    train_label = np.array([1] * 1484 + [0] * 1484).astype(np.float32)
    train_label = to_categorical(train_label, num_classes=2)

    # Read the testing set
    test_pos_seqs = np.array(read_fasta('../data/layer1/test_enhancers.fa'))
    test_neg_seqs = np.array(read_fasta('../data/layer1/test_nonenhancers.fa'))

    test_seqs = np.concatenate((test_pos_seqs, test_neg_seqs), axis=0)

    test_onehot = np.array(to_one_hot(test_seqs)).astype(np.float32)
    test_properties_code = np.array(to_properties_code(test_seqs)).astype(np.float32)

    test = np.concatenate((test_onehot, test_properties_code), axis=1)


    test_label = np.array([1] * 200 + [0] * 200).astype(np.float32)
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



            BATCH_SIZE = 30
            EPOCHS = 300



            model = load_model('../models/one_enhancer_model_' + str(fold_count) + '.h5')


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

    plt.title('ROC Curve of First Layer')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('../images/ROC_Curve_of_First_Layer.jpg',dpi=1200, bbox_inches='tight')
    plt.legend(loc='lower right')
    plt.show()


