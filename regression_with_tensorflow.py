# -*- coding: utf-8 -*-
"""Regression-with-Tensorflow.ipynb


Original file is located at
    https://colab.research.google.com/drive/1vb7cKBqyfHDol1Qqp2o3et_UziixU97C
"""

import tensorflow as tf
print(tf.__version__)

#creating data to view and fit
import numpy as np
import matplotlib.pyplot as plt
x=np.array([-7.0,-4.0,-1.0,2.0,5.0,8.0,11.0,14.0])

#creating labels
y=np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])
plt.scatter(x,y)

#input and output Tensor
house_info = tf.constant(["bedroom","bathroom","garage"])
house_price = tf.constant([939700])
house_info,house_price

x=tf.constant(x)
y-tf.constant(y)

#steps in modelling with Tensorflow
# 1. defining the model like defining the input,output and hidden layers
# 2. compiling a model : Loss function and optimizer and evaluation metrics
# 3. Fitting a model

tf.random.set_seed(42)
#1
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

#2
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics = ["mae"]
              )

#3
model.fit(tf.expand_dims(x, axis=-1), y, epochs=5)

model.predict([17.0])

"""# Improving Our Model

##We can improve our model by following methods:
1. **creating a model** -> here we can add more layers,increase the number of hidden units,change activation function of each layer.
2. **Compiling a model** -> here we might change the optimization function or perhaps **Learning Rate** of optimization function.
3. **Fitting a Model** -> here we might fit model for more **epochs** or more data.
"""

tf.random.set_seed(42)
#1
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

#2
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics = ["mae"]
              )

#3
model.fit(tf.expand_dims(x, axis=-1), y, epochs=100)

model.predict([17.0])

tf.random.set_seed(42)
#1
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100,activation= "relu"),
    tf.keras.layers.Dense(1)
])

#2
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics = ["mae"]
              )

#3
model.fit(tf.expand_dims(x, axis=-1), y, epochs=100)

model.predict([17.0])

tf.random.set_seed(42)
#1
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50,activation = None),
    tf.keras.layers.Dense(1)
])

#2
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.Adam(),
              metrics = ["mae"]
              )

#3
model.fit(tf.expand_dims(x, axis=-1), y, epochs=100)

model.predict([17.0])

tf.random.set_seed(42)
#1
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50,activation = None),
    tf.keras.layers.Dense(1)
])

#2
model.compile(loss="mae",
              optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics = ["mae"]
              )

#3
model.fit(tf.expand_dims(x, axis=-1), y, epochs=100)

model.predict([17.0])

"""# Evaluating a model preformance"""

X=tf.range(-100,100,4)
X

Y = X + 10
Y

import matplotlib.pyplot as plt
plt.scatter(X,Y)

len(X)

#Splitting_the _data
X_train=X[:40]
Y_train=Y[:40]
X_test = X[40:]
Y_test = Y[40:]
len(X_train),len(X_test),len(Y_train),len(Y_test)

plt.figure(figsize=(7,7))
plt.scatter(X_train,Y_train,c="b",label="training_data")

plt.scatter(X_test,Y_test,c="g",label="testing data")
plt.legend()

tf.random.set_seed(42)
#1
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

#2
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics = ["mae"]
              )

#3
model.fit(tf.expand_dims(X_train, axis=-1), Y_train, epochs=100)

model.summary()

tf.random.set_seed(42)
#1
model = tf.keras.Sequential([
    tf.keras.layers.Dense(3,input_shape = [1])
])

#2
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics = ["mae"]
              )

model.summary()

model.fit(tf.expand_dims(X_train, axis=-1), Y_train, epochs=100,verbose=0)

from tensorflow.keras.utils import *
plot_model(model=model,show_shapes = True)

Y_pred =model.predict(X_test)
Y_pred

import matplotlib.pyplot as plt

def plot_predictions(
    train_data=X_train,
    train_labels = Y_train,
    test_data = X_test,
    test_labels = Y_test,
    predictions = Y_pred
):
 plt.figure(figsize=(10,7))

 plt.scatter(train_data,train_labels,c="b",label="training data")


 plt.scatter(test_data,test_labels,c="g",label="testing data")

 plt.scatter(test_data,predictions,c="r",labels = "Predictions")

 plt.legend();

 plt.show()

plot_predictions()

