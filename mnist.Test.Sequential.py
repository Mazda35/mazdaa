
from keras.datasets import mnist
import numpy as np

# Load data --------------------------------------
(trainImages,trainLabels),(testImages,testLabels)=mnist.load_data()

# Data attributes
# print('train images dimensions : ',trainImages.ndim)
# print('train images shape : ',trainImages.shape)
# print('train images type : ',trainImages.dtype)

# Plot Function ---------------------------------------
def plotHistory(netHistory):
    import matplotlib.pyplot as plt
    history=netHistory.history
    losses=history['loss']
    accuracies=history['acc']
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.plot(losses)
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracies')
    plt.plot(accuracies)

# preprocesing -----------------------------------------
xtrain=trainImages.reshape(60000,784)
xtest=testImages.reshape(10000,784)

xtrain=xtrain.astype('float32')
xtest=xtest.astype('float32')
xtrain/=255
xtest/=255

from keras.utils import np_utils
ytrain=np_utils.to_categorical(trainLabels)
ytest=np_utils.to_categorical(testLabels)

# Creating model -------------------------------
from keras.models import Sequential
from keras.layers import Dense,Dropout
model=Sequential()
model.add(Dense(500,activation='relu',input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))

# model.summary()
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
model.compile(optimizer=SGD(lr=0.001), loss=categorical_crossentropy, metrics=['accuracy'])

# Model training -------------------------------------
network=model.fit(xtrain,ytrain,batch_size=128,epochs=2)
history=network.history
type(history)

# Evaluation ----------------------------------------
model.evaluate()

# --------------------
import graphviz



