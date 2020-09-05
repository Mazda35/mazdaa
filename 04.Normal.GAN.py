
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.layers import Dense,Flatten,Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam

# Image dimensions
imgRows=28
imgCols=28
imgChannels=1

imageShape=(imgRows,imgCols,imgChannels)

zDim=100

########################### Function Building

# Generative Function
def generativeFn(imageShape,zDim):
    model=Sequential()
    model.add(Dense(128,input_dim=zDim))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(28*28*1,activation='tanh'))
    model.add(Reshape(imageShape))
    return model

# Dicriminative Function
def discriminativeFn(imageShape):
    model=Sequential()
    model.add(Flatten(input_shape=imageShape))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1,activation='sigmoid'))
    return model

# GAN Function(model)
def ganFn(generative,discriminative):
    model=Sequential()
    model.add(generative)
    model.add(discriminative)
    return model

########################### Required Parameters

# Discriminative Network
disVariable=discriminativeFn(imageShape)
disVariable.compile(loss='binary_crossentropy',optimizer=Adam(),metrics=['accuracy'])

# Generative Network
genVariable=generativeFn(imageShape,zDim)
disVariable.trainable=False

# GAN Network
ganVariable=ganFn(genVariable,disVariable)
ganVariable.compile(loss='binary_crossentropy',optimizer=Adam())

# Variables needed for train
losses=[]
accuracies=[]
iterationChecks=[]

# Train Function
def trainFn(iterations,batchSize,interval):
    (xtrain,_),(_,_)=mnist.load_data()
    xtrain=xtrain/127.5-1
    xtrain=np.expand_dims(xtrain,axis=3)

    real=np.ones((batchSize,1))
    fake=np.zeros((batchSize,1))

    for iteration in range(iterations):
        # Discriminate function training
        # Real images
        ids=np.random.randint(0,xtrain.shape[0],batchSize)
        imgs=xtrain[ids]
        # Fake images
        z=np.random.normal(0,1,(batchSize,100))
        imgsGenerated=genVariable.predict(z)

        disLossReal=disVariable.train_on_batch(imgs,real)
        disLossFake=disVariable.train_on_batch(imgsGenerated,fake)

        dLoss,accuracy=0.5*np.add(disLossReal,disLossFake)

        # Generative Function training
        z = np.random.normal(0, 1, (batchSize, 100))
        genLoss=ganVariable.train_on_batch(z,real)

        #Outputs
        if (iteration+1)%interval==0:
            losses.append((dLoss,genLoss))
            accuracies.append(100.0*accuracy)
            iterationChecks.append(iteration+1)

            print("%d [D loss: %f , acc: %.2f] [G loss: %f]" %
                  (iteration + 1, dLoss, 100.0 * accuracy, genLoss))
            showImage(genVariable)

def showImage(gen):
    z = np.random.normal(0, 1, (16, 100))
    imgGenerated=gen.predict(z)
    imgGenerated=0.5*imgGenerated+0.5

    fig,axs=plt.subplots(4,4,figsize=(4,4),sharex=True,sharey=True)
    count=0
    for i in range(4):
        for j in range(4):
            axs[i,j].imshow(imgGenerated[count,:,:,0],cmap='Greys')
            axs[i,j].axis('off')
            count+=1

    fig.show()


trainFn(5000,128,1000)









