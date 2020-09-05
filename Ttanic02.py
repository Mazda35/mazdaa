import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import SGDClassifier
import sklearn
df=pd.read_csv('C:/Users/zigorat/Desktop/Kaggle/Titanic/train.csv')
df_test=pd.read_csv('C:/Users/zigorat/Desktop/Kaggle/Titanic/test.csv')
# ytest=pd.read_csv('C:/Users/zigorat/Desktop/Kaggle/Titanic/gender_submission.csv')
# ytest=ytest.drop(['PassengerId'],axis=1)
#    IN THIS FUNCTION WE SPLIT NAME INTO WORDS
def FuncName(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'Unknown'
#     IN THIS FUNCTION WE WANT TO CATEGORISE TITLES
def FuncG(tt):
    if tt in ['Mr']:
        return 1
    elif tt in ['Master']:
        return 3
    elif tt in ['Ms','Mlle','Miss']:
        return 4
    elif tt in ['Mrs','Mme']:
        return 5
    else:
        return 2
# ADD TITLE COLUMN IN ORDER TO CATEGORIZE PASSENGER ID COLUMN
df['title']=df['Name'].apply(FuncName).apply(FuncG)
df_test['title']=df_test['Name'].apply(FuncName).apply(FuncG)
# THEN WE DROP UNNECESSARY COLUMNS
df=df.drop(['PassengerId','Name','Ticket'],axis=1)
df['title']=df['title'].astype(float)
# df_test['title']=df_test.drop(['Name','Ticket'],axis=1)
df_test['title']=df_test['title'].astype(float)
# HERE WE USE DUMMIES FUNCTION TO CATEGORISE EMBARK COLUMN IN 3 DIFFERENT COLUMNS DUE TO SIMPLIFY CALCULATIONS
edt=pd.get_dummies(df['Embarked'])
edt.drop(['S'],axis=1,inplace=True)     #Why do this?Check it later
edt_test=pd.get_dummies(df_test['Embarked'])
edt_test.drop(['S'],axis=1,inplace=True)
df=df.join(edt)
df_test=df_test.join(edt_test)
df.drop(["Embarked"],axis=1,inplace=True)
df_test.drop(["Embarked"],axis=1,inplace=True)
# IN FARE COLUMN WE FILL EMPTY CELLS WITH MEDIAN
df_test['Fare'].fillna(df_test['Fare'].median(),inplace=True)
# AGE IMPUTE
df['Age']=df.groupby(['Pclass'])['Age'].transform(lambda x:x.fillna(x.mean()))
df_test['Age']=df_test.groupby(['Pclass'])['Age'].transform(lambda x:x.fillna(x.mean()))
# CABIN
# iT HAS A LOT OF NaN VALUES, SO IT WONT CAUSE REMARKABLE IMPACT ON PREDICTION
df.drop('Cabin',axis=1,inplace=True)
df_test.drop('Cabin',axis=1,inplace=True)
#SEX
s=sorted(df['Sex'].unique())
z=zip(s,range(0,len(s)+1))
gm=dict(z)
df['Sex']=df['Sex'].map(gm).astype(int)
df_test['Sex']=df_test['Sex'].map(gm).astype(int)
# HERE WE HAVE TO CHOOSE SURVIVED COLUMN AS TARGET SO WE HAVE:
xtrain=df.drop('Survived',axis=1).astype(int)
# print(xtrain)
ytrain=df['Survived'].astype(int)
# print(ytrain)
# HERE WE CREATE MODEL
model=LogisticRegression()
model.fit(xtrain,ytrain)
pop=model.score(xtrain,ytrain)
xtest=df_test.drop(['PassengerId','Name','Ticket'],axis=1)
#
# sgd =SGDClassifier(max_iter=5, tol=None)
# sgd.fit(xtrain,ytrain)
# ypred= sgd.predict(xtest)
#
# pop=sgd.score(xtrain, ytrain)


# print(oop)
print(pop)
# print(xtest)
# print(xtrain)
ypred=model.predict(xtest)
# ytest['Predict']=ypred.astype(int)
# ytest['Predict']=ytest.drop(['PassengerId'],axis=1)
print(ypred)
# print(ytest.columns)
# accuracy=0

# print(ytest)
# for i,row in ytest.iterrows():
#
#     if row['Survived']==row['Predict']:
#         accuracy+=1
#     else:
#         print('not')
aaa=pd.DataFrame(ypred)
# ytest.to_csv('C:/Users/zigorat/Desktop/Kaggle/Titanic/Result.csv')
aaa.to_csv('C:/Users/zigorat/Desktop/Kaggle/Titanic/aaa.csv')
# print(accuracy)







