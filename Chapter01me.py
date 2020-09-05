
import numpy as np
features=['BreadN','Milk','Cheese','Apple','Banana']
X= np.loadtxt("C:/Users/zigorat/Desktop/Python Projects/Learning Data Mining with python/Learning-Data-Mining-with-Python-master/Chapter 1/affinity_dataset.txt")

# print(X)

from collections import defaultdict
validRules=defaultdict(int)
invalidRules=defaultdict(int)
numOccurances=defaultdict(int)
nFeatures=len(features)

print(nFeatures)

for sample in X:
    for premise in range(5):
        for conclusion in range(5):
            if sample[int(premise)]==1 and sample[int(conclusion)]==1:
                validRules[(premise,conclusion)]+=1
                if premise==conclusion:
                   numOccurances[(premise,conclusion)]+=1
            if sample[int(premise)]==0 and sample[int(conclusion)]==1:invalidRules[(premise,conclusion)]+=1
support=validRules
print('numOccurances: ',numOccurances)
print('support: ',support)
print('invalidRules',invalidRules)
print(validRules.keys())
print(numOccurances.keys())
confidence=defaultdict(float)
for premise,conclusion in validRules.keys():
    rule=(premise,conclusion)
    confidence[rule]=validRules[rule]/validRules[(premise,premise)]
print(confidence)

def PrintRule(premise,conclusion,support,confidence,features):
    premiseName=features[premise]
    conclusionName=features[conclusion]
    print('Rule: if a person buys {0} they will buy {1} also'.format(premiseName,conclusionName),
          '\n Support: ',support[(premise,conclusion)],
          '\n Confidence : ',confidence[(premise,conclusion)])

print(PrintRule(1,3,support,confidence,features))

























