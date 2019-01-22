from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import traffic_data.read_data as read_data
from gcn_kl.utils import *
from gcn_kl.models import GCN, MLP
from collections import Counter

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 500, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 8, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 2, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('test_num', 456, 'Number of test nodes.')

# Load data
# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
adj, features0, y_train1, _, _, _, _, test_mask0 = read_data.load_data_traffic(week=0, test_num=FLAGS.test_num, hour=0)
# Some preprocessing
# features0 = preprocess_features(features0)
features0 = sparse_to_tuple(features0)
support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features0[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train1.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features0[2][1], logging=True)

# Initialize session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# sess = tf.Session(config=config)


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.pppp], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2]


weeks = 11  # T = weeks
window_slide = 43 - weeks  # window size
opinion_e = []
opinion_x = np.zeros_like(y_train1)
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for j in range(6)[2:3]:
        for w in range(window_slide)[:1]:
            error_opinion_p = 0.0  # initial
            for k in range(25):
                # Train model
                t1 = time.time()
                for epoch in range(weeks):
                    # more example
                    # p = np.mod(epoch, weeks) + j
                    p = np.mod(epoch, weeks) + w
                    _, features, y_train, _, _, train_mask, _, _ = read_data.load_data_traffic(week=p,
                                                                                               test_num=FLAGS.test_num,
                                                                                               hour=j)
                    # features = preprocess_features(features)
                    features = sparse_to_tuple(features)
                    t = time.time()
                    # Construct feed dictionary
                    if k > 20:
                        y_train = y_train + opinion_x  # give the soft label
                        train_mask = np.ones_like(train_mask, dtype=bool)
                    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
                    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

                    # Training step
                    # outs = sess.run([model.opt_op, model.loss, model.accuracy, model.predict()], feed_dict=feed_dict)
                    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.pppp], feed_dict=feed_dict)

                # Testing
                cost_test = []
                acc_test = []
                test_prediction = []
                y_truth = []
                for i in range(weeks):
                    i = i + w
                    _, features, _, _, y_test, _, _, test_mask = read_data.load_data_bigdata(week=i,
                                                                                             test_num=FLAGS.test_num,
                                                                                             hour=j)
                    # features = preprocess_features(features)
                    features = sparse_to_tuple(features)
                    test_cost, test_acc, test_duration, prediction = evaluate(features, support, y_test, test_mask,
                                                                              placeholders)
                    cost_test.append(test_cost)
                    acc_test.append(test_acc)
                    test_prediction.append(prediction)
                    y_truth.append(y_test)
                cost_test = np.mean(cost_test)
                acc_test = np.mean(acc_test)
                opinion_gcn = read_data.opinion_c(test_prediction)
                opinion_truth = read_data.opinion_c(y_truth)
                error_opinion = read_data.opinion_error(opinion_gcn, opinion_truth, test_mask0)
                print("Iteration:", '%02d' % k, "accuracy=", "{:.5f}".format(acc_test * 100), "GCN opinion error=",
                      "{:.5f}".format(error_opinion))
                # opinion_e.append(error_opinion)
                np.save("./result/opinion_0.3_ph_T11_gcn_6hour_test.npy", opinion_e)

                t2 = time.time()
                # KL-divergence to get opinion
                opinion_p, opinion_l = read_data.get_opinion(test_prediction, test_mask0)
                opinion_x = opinion_l
                error_opinion_p = read_data.opinion_error(opinion_p, opinion_truth, test_mask0)
                print("GCN-KL opinion error=", "{:.5f}".format(error_opinion_p))
            opinion_e.append(error_opinion_p)

print("weeks:", '%02d' % weeks, "test num:", '%03d' % FLAGS.test_num, "Mean opinion error=",
      "{:.5f}".format(np.mean(opinion_e)))
