---
layout: post
title: Learn and use ML
---

고수준의 Keras API는 deep learning model들을 생성하고 훈련하기 위한 블록 제작을 제공합니다. 
처음 시작하는 분을 위한 예제를 시작하고, [TensorFlow Keras guide](https://www.tensorflow.org/guide/keras)를 읽으세요.

1. [Basic classification](https://www.tensorflow.org/tutorials/keras/basic_classification)
2. [Text classification](https://www.tensorflow.org/tutorials/keras/basic_text_classification)
3. [Regression](https://www.tensorflow.org/tutorials/keras/basic_regression)
4. [Overfitting and underfitting](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit)
5. [Save and load](https://www.tensorflow.org/tutorials/keras/save_and_restore_models)


{% highlight python3 %}

**import** tensorflow **as** tf
mnist = tf.keras.datasets.mnist

*#Download MNIST data, require internet*
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

*#Configuration Model*
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

*#Compile Configuration Model*
model.compile(optimizer = ‘Adam’,
              loss = ‘sparse_categorical_crossentropy’,
              metrics= [‘accuracy’])

*#Run training*
model.fit(x_train, y_train, epochs = 5)
*#Run test data*
model.evaluate(x_test, y_test)

{%endhighlight%}
