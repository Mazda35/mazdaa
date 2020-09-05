
import pandas_datareader.data as rd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pg=rd.DataReader('PG',data_source='yahoo',start='1995-1-1')
# pg.to_csv('pg.csv')
pg=pd.read_csv('pg.csv')
# print(pg)

pg['singleReturn']=(pg['Adj Close']/pg['Adj Close'].shift(1))-1
pg['logReturn']=np.log(pg['Adj Close']/pg['Adj Close'].shift(1))
# print(pg['singleReturn'])

avrageReturn=pg['singleReturn'].mean()
rorSum=pg['singleReturn'].sum()
print(avrageReturn)
print(rorSum)

pg.to_csv('pg2.csv')
# plt.plot(pg['singleReturn'],color='blue')
# plt.figure()
# plt.plot(pg['logReturn'],color='purple')
# plt.show()

# ------------------------------------------------------------
# portfolio=['PG','MSFT','F','GE']
# myData=pd.DataFrame()
# for t in portfolio:
#     myData[t]=rd.DataReader(t,data_source='yahoo',start='1995-1-1')['Adj Close']

# print(myData)
# myData.to_csv('myData.csv')
myData=pd.read_csv('myData.csv')
# print(myData.iloc[[0],[1]])
myData=myData.iloc[:,1:]
print(myData.iloc[0])
newData=pd.DataFrame()
for n in myData:
    newData=(myData/(myData.iloc[0])*100)
print(newData)

plt.plot(newData)
plt.figure()
plt.plot(myData)
plt.show()

returns=(newData/newData.shift(1))-1
# print(returns)
weights1=np.array([0.25,0.25,0.25,0.25])
weights2=np.array([0.4,0.4,0.15,0.05])
annualReturns=returns.mean()*250
print(annualReturns)
portfolio1=str(round(np.dot(annualReturns,weights1),5)*100)+'%'
portfolio2=str(round(np.dot(annualReturns,weights2),5)*100)+'%'
print(portfolio1)
print(portfolio2)


