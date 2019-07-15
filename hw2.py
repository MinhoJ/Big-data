import tensorflow as tf
import numpy as np
import pandas as pd

tf.set_random_seed(777)  # for reproducibility

substitution = {
    'Fair': 0,
    'Good': 1,
    'Ideal': 2,
    "Premium": 3,
    "Very Good": 4
}

csv = pd.read_csv('diamonds.csv')
x_data = csv[["carat", "depth", "table", "price"]]
# y_data = pd.get_dummies(csv.cut)
y_data = list(csv["cut"].apply(lambda x: substitution[x]).to_csv(header=None, index=False).replace("\r", "").replace("\n",
                                                                                                                ""))
y_data = np.array(y_data)
print(y_data)
# y_data = pd.DataFrame({'cut': y_data})
print(x_data.shape, y_data.shape)

X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.int32, [None])
nb_classes = 5

Y_one_hot = tf.one_hot(Y, nb_classes)
print(Y_one_hot)

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                    labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# 결과를 어느클래스인가를 표시 (0~6까지의 값을 갖는 벡터)
prediction = tf.math.argmax(hypothesis, 1)
# Y_one_hot을 argmax를 이용하여 다시 0~6의 값으로 변환한 후
# prediction과 같은지 비교함
correct_prediction = tf.equal(prediction, tf.math.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={
                X: x_data, Y: y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                step, loss, acc))
    # Accuracy report
    p, a, y = sess.run([prediction, accuracy, Y],
                       feed_dict={X: x_data, Y: y_data})

    print("\nPredcited: \n", p)
    print("\nY: \n", y)

    print("\nAccuracy: \n", a)
