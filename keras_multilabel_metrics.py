from keras import backend as K
import tensorflow as tf
from keras.optimizers import Adam
from keras import Input, Model

def recall_m(y_true, y_pred):
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)
    pair_sum = tf.add(y_true, y_pred)
    true_ones = K.equal(pair_sum, 2.)
    true_positives = K.cast(true_ones, K.floatx())
    true_positive_count = K.sum(true_positives)
    label_positive_count = K.sum(y_true)
    recall = true_positive_count / label_positive_count
    return recall


def precision_m(y_true, y_pred):
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)
    pair_sum = tf.add(y_true, y_pred)
    true_ones = K.equal(pair_sum, 2.)
    true_positives = K.cast(true_ones, K.floatx())
    true_positive_count = K.sum(true_positives)
    pred_positive_count = K.sum(y_pred)
    precision = true_positive_count / pred_positive_count
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall))


# Another way to define your optimizer
adam = Adam(lr=0.001)
# We add metrics to get more results you want to see
model = Model(inputs="your inputs")
model.compile(optimizer=adam, loss="mean_squared_error", metrics=['categorical_accuracy', recall_m, precision_m, f1_m])
