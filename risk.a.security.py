
import pandas_datareader.data as rd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# portfolio=['PG','BEI.DE']
# data=pd.DataFrame()
# for t in portfolio:
#     data[t]=rd.DataReader(t,data_source='yahoo',start='2007-1-1',end='2017-3-23')['Adj Close']
#
# data.to_csv('risk a security.csv')

data=pd.read_csv('risk a security.csv')
data=data.set_index('Date')
print(data)
returns=np.log(data/data.shift(1))
print(returns)

returnMean=returns.mean()*250
returnStd=returns.std()*250**0.5

# a=returns[['PG','BEI.DE']].mean()*250
# b=returns[['PG','BEI.DE']].std()*250**0.5
print(returnMean,'\t\t',returnStd)
# print(a,'\t\t',b)


pgVar=returns['PG'].var()
pgVarAnnual=returns['PG'].var()*250
beiVar=returns['BEI.DE'].var()
beiVarAnnual=returns['BEI.DE'].var()*250
print('PG Variance : ',pgVar,'\tAnnual : ',pgVarAnnual)
print('BEI Variance : ',beiVar,'\tAnnual : ',beiVarAnnual)

covMatrix=returns.cov()
corMatrix=returns.corr()
print(covMatrix)
print(corMatrix)

