
import numpy as np
import matplotlib.pyplot as plt


from keras.datasets import mnist
from keras.layers import Dense,Flatten,Reshape,BatchNormalization,Activation
from keras.layers import Dropout,Lambda
from keras.layers.convolutional import Conv2D,Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as bk
from keras.utils import to_categorical

# Class for inputs
class DataSet:

    def __init__(self,labeledNumber):
        self.labeledNumber=labeledNumber
        (self.xtrain,self.ytrain),(self.xtest,self.ytest)=mnist.load_data()

        def imagePreprocessing(x):
            x=(x.astype(np.float)-127.5)/127.5    #data will transform from 0-255 to -1,1
            x=np.expand_dims(x,axis=3)
            return x

        def labelPreprocessing(y):
            return y.reshape(-1,1)       #here we add a dimension to the label dataset
                                         # first dimension will remain as before and one dimension will add to the it
                                         # 60000 will transform to (60000,1)
        # now we prepeoces xtrain, xtest, ytrain and y test
        self.xtrain=imagePreprocessing(self.xtrain)
        self.xtest=imagePreprocessing(self.xtest)
        self.ytrain=labelPreprocessing(self.ytrain)
        self.ytest=labelPreprocessing(self.ytest)

    # we read labeledNumber data as label data for example 100 entry here
    def readBatchLabeled(self,batchSize):
        ids=np.random.randint(0,self.labeledNumber,batchSize)
        images=self.xtrain[ids]
        labels=self.ytrain[ids]
        return images,labels

    # then put remain data as unlabeled data other 59900 here
    def readBatchUnlabeled(self,batchSize):
        ids=np.random.randint(self.labeledNumber,self.xtrain.shape[0],batchSize)
        images=self.xtrain[ids]
        labels=self.ytrain[ids]
        return images,labels

    # we read labeledNumber data for xtrain and ytrain
    def readTrainingData(self):
        xtrain=self.xtrain[range(self.labeledNumber)]
        ytrain=self.ytrain[range(self.labeledNumber)]
        return xtrain,ytrain

    # here we dont care about labeled data and just read 10000 test from all 70000
    def readTestingData(self):
        return self.xtest,self.ytest


labeledNumber=100

dataset=DataSet(labeledNumber)

# Image dimensions
imgRows=28
imgCols=28
imgChannels=1

imageShape=(imgRows,imgCols,imgChannels)

zDim=100
numClasses=10

########################### Function Building

# Generative Function
def generativeFn(zDim):
    model=Sequential()
    model.add(Dense(256*7*7,input_dim=zDim))
    model.add(Reshape((7,7,256)))

    # First Transposed Concolutional (ouput: 14*14*128)
    model.add(Conv2DTranspose(128,kernel_size=3,strides=2,padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    # Second Transposed Concolutional (ouput: 14*14*64)
    model.add(Conv2DTranspose(64,kernel_size=3,strides=1,padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    # Last Transposed Convolutional (ouput: 28*28*1)
    model.add(Conv2DTranspose(1,kernel_size=3,strides=2,padding='same'))

    # Activation
    model.add(Activation('tanh'))

    return model

# Dicriminative Function
def discriminativeFn(imageShape):
    model=Sequential()

    # First convolutional (ouput: 14*14*32)
    model.add(Conv2D(32,kernel_size=3,strides=2,input_shape=imageShape,padding='same'))
    model.add(LeakyReLU(alpha=0.01))

    # Second convolutional (ouput: 7*7*64)
    model.add(Conv2D(64,kernel_size=3,strides=2,input_shape=imageShape,padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    # last convolutional (ouput: 3*3*128)
    model.add(Conv2D(128,kernel_size=3,strides=2,input_shape=imageShape,padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    model.add(Dropout(0.5))

    model.add(Flatten())

    # model.add(Dense(1, activation='sigmoid'))
    # Here we use last line insteas of line 125
    model.add(Dense(numClasses))
    return model

# Supervised Network
def discriminativeSupervisedFn(disNetwork):
    model=Sequential()
    model.add(disNetwork)
    model.add(Activation('softmax'))
    return model

# UnSupervised Network
def discriminativeUnsupervisedFn(disNetwork):
    model=Sequential()
    model.add(disNetwork)

    # Here we want to distinguish between real and fake input
    # meaning real is ten classes and fake is one additional class
    def predict(x):
        prediction=1.0-(1.0/(bk.sum(bk.exp(x),axis=-1,keepdims=True)+1.0))
        return prediction

    model.add(Lambda(predict))
    return model

# GAN Function(model)
def ganFn(generative,discriminative):
    model=Sequential()
    model.add(generative)
    model.add(discriminative)
    return model

########################### Organising network
########################### Required Parameters

# Discriminative Network
disVariable=discriminativeFn(imageShape)

disVariableSupervised=discriminativeSupervisedFn(disVariable)
disVariableSupervised.compile(loss='categorical_crossentropy',
                              optimizer=Adam(),metrics=['accuracy'])

disVariableUnupervised=discriminativeSupervisedFn(disVariable)
disVariableUnupervised.compile(loss='binary_crossentropy',
                               optimizer=Adam())


# Generative Network
genVariable=generativeFn(zDim)
disVariableUnupervised.trainable=False  #weights are with no change in gen network training

# GAN Network
ganVariable=ganFn(genVariable,disVariableUnupervised)
ganVariable.compile(loss='binary_crossentropy',optimizer=Adam())

# Variables needed for train
supervisedLosses=[]
iterationChecks=[]

# Train Function
def trainFn(iterations,batchSize,interval):

    real=np.ones((batchSize,1))
    fake=np.zeros((batchSize,1))

    for iteration in range(iterations):
        # Discriminate function training
        # Real images
        imgsLabeled,labels=dataset.readBatchLabeled(batchSize)
        labels=to_categorical(labels,num_classes=numClasses)

        imgsUnlabeled=dataset.readBatchUnlabeled(batchSize)

        # Fake images
        z=np.random.normal(0,1,(batchSize,100))
        imgsGenerated=genVariable.predict(z)

        # Supervised Loss
        dLossSupervised,accuracy=disVariableSupervised.train_on_batch(imgsLabeled,labels)

        # Unsupervised Loss
        dLossReal=disVariableUnupervised.train_on_batch(imgsUnlabeled,real)
        dLossFake=disVariableUnupervised.train_on_batch(imgsUnlabeled,fake)

        dLossUnsupervised=0.5*np.add(dLossReal,dLossFake)

        # Generative Function training
        z = np.random.normal(0, 1, (batchSize, 100))
        genLoss=ganVariable.train_on_batch(z,real)

        #Outputs
        if (iteration+1)%interval==0:
            supervisedLosses.append(dLossSupervised)
            iterationChecks.append(iteration+1)

            print("%d [Supervised loss: %.4f , acc: %.2f] [Unsupervised loss: %f]" %
                  (iteration + 1, dLossSupervised, 100.0 * accuracy, dLossUnsupervised))



trainFn(3000,32,800)


############################TEST TEST TEST##############################

x,y=dataset.readTrainingData()
y=to_categorical(y,num_classes=numClasses)

_,accuracy=disVariableSupervised.evaluate(x,y)
print("Training accuracy: %.2f]" %
                  (100.0 * accuracy))

x,y=dataset.readTestingData()
y=to_categorical(y,num_classes=numClasses)

_,accuracy=disVariableSupervised.evaluate(x,y)
print("Testing accuracy: %.2f]" %
                  (100.0 * accuracy))


########################Comparison between Semi-Supervised and Supervised Networks #############

# Supervised Network
mnistClassifier= discriminativeSupervisedFn(discriminativeFn(imageShape))
mnistClassifier.compile(loss='categorical_crossentropy',
                        optimizer=Adam(),
                        metrics=['accuracy'])

imgs,labels=dataset.readTrainingData()
labels=to_categorical(labels,num_classes=numClasses)

training=mnistClassifier.fit(x=imgs,y=labels,batch_size=32,epochs=30,verbose=1)

x,y=dataset.readTrainingData()
y=to_categorical(y,num_classes=numClasses)

_,accuracy=mnistClassifier.evaluate(x,y)
print("Training accuracy: %.2f]" %(100.0 * accuracy))

x,y=dataset.readTestingData()
y=to_categorical(y,num_classes=numClasses)

_,accuracy=mnistClassifier.evaluate(x,y)
print("Testing accuracy: %.2f]" %(100.0 * accuracy))







