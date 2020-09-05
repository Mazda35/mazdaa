
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
xtrain=trainImages.reshape(60000,28,28,1)
xtest=testImages.reshape(10000,28,28,1)

xtrain=xtrain.astype('float32')
xtest=xtest.astype('float32')
xtrain/=255
xtest/=255

from keras.utils import np_utils
ytrain=np_utils.to_categorical(trainLabels)
ytest=np_utils.to_categorical(testLabels)

# Creating model -------------------------------
from keras.models import Model
from keras.layers import Conv2D,MaxPooling2D,Input,Flatten,Dense
import keras
input=Input(shape=(28,28,1))
conv1=Conv2D(16,(3,3),activation='relu',padding='same',strides=2)(input)
# pool1=MaxPooling2D(pool_size=2)(conv1)
conv2=Conv2D(32,(3,3),activation='relu',padding='same',strides=2)(conv1)
# pool2=MaxPooling2D(pool_size=2)(conv2)
flat=Flatten()(conv2)
outLayer=Dense(10,activation='softmax')(flat)
model=Model(input,outLayer)

# model.summary()
from keras.losses import categorical_crossentropy
model.summary()
model.compile(optimizer=keras.optimizers.Adam(),loss=keras.losses.categorical_crossentropy,metrics=['accuracy'])

# Model training -------------------------------------
network=model.fit(xtrain,ytrain,batch_size=128,epochs=2)
plotHistory(network)

# Evaluation ----------------------------------------
testLoss,testAcc=model.evaluate(xtest,ytest)
testLAbelPredict= model.predict(xtest)
import numpy as np
testLAbelPredict = np.argmax(testLAbelPredict, axis=1)





