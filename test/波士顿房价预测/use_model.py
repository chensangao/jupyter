# coding: utf-8
import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# get data
boston = load_boston()
X = boston.data
y = boston.target

# split train and test data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=0)

# scale data
X_test = scale(X_test)
y_test = scale(y_test).reshape((-1,1))

def predict(X,y,keep_prob):

    with tf.Session() as sess:

        # restore saver
        saver = tf.train.import_meta_graph(meta_graph_or_file="nn_boston_model/nn_boston.model-5000.meta")
        model_file = tf.train.latest_checkpoint(checkpoint_dir="nn_boston_model")
        saver.restore(sess=sess,save_path=model_file)

        # init graph
        graph = tf.get_default_graph()

        # get placeholder from graph
        xs = graph.get_tensor_by_name("inputs:0")
        ys = graph.get_tensor_by_name("y_true:0")
        keep_prob_s = graph.get_tensor_by_name("keep_prob:0")

        # get operation from graph
        pred = graph.get_tensor_by_name("pred:0")

        # run pred
        feed_dict = {xs: X, ys: y, keep_prob_s: keep_prob}
        y_pred = sess.run(pred,feed_dict=feed_dict)

    return y_pred.reshape(-1)

y_pred = predict(X=X_test,y=y_test,keep_prob=1)

# show data
plt.plot(range(len(y_test)),y_test,'b')
plt.plot(range(len(y_pred)),y_pred,'r--')
plt.show()

