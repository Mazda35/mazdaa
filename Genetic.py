

import random
import datetime
import time
import statistics
import sys


class Chromosome:
    Genes=None
    Fitness=None

    def __init__(self,genes,fitness):
        self.Genes=genes
        self.Fitness=fitness


def _generateParent(length, geneSet, getFitness):
    genes=[]
    while len(genes)<length:
        sampleSize=min(length-len(genes), len(geneSet))
        genes.extend(random.sample(geneSet, sampleSize))
    fitness=getFitness(genes)
    return Chromosome(genes,fitness)

def _mutate(parent, geneSet,getFitness):
    index = random.randrange(0, len(parent.Genes))
    childGenes=parent.Genes[:]
    newGene, alternate=random.sample(geneSet, 2)
    childGenes[index]= alternate \
        if newGene==childGenes[index]\
        else newGene
    fitness = getFitness(childGenes)
    return Chromosome(childGenes, fitness)

def getImprovement(newChild,generateParent):
    bestParent = generateParent()
    yield bestParent
    while True:
        child = newChild(bestParent)
        if bestParent.Fitness> child.Fitness:
            continue
        if not child.Fitness > bestParent.Fitness:
            bestParent=child
            continue
        yield child
        bestParent = child



def getBest(getFitness, targetLen, optimalFitness, geneSet, display):
    random.seed()

    def fnMutate(parent):
        return _mutate(parent,geneSet,getFitness)

    def fnGenerateParent():
        return _generateParent(targetLen,geneSet,getFitness)

    for improvement in getImprovement(fnMutate,fnGenerateParent):
       display(improvement)
       if not optimalFitness > improvement.Fitness:
           return improvement

# if not optimalFitness > child.Fitness:
#     return child



class Benchmark:
    @staticmethod
    def Run(function):
        timings=[]
        stdout=sys.stdout
        for i in range(100):
            sys.stdout=None
            startTime=time.time()
            function()
            seconds=time.time()-startTime
            sys.stdout=stdout
            timings.append(seconds)
            mean=statistics.mean(timings)
            if i<10 or i%10==9:
                print('{0} {1:3.2f} {2:3.2f}'.format(1+i,mean,statistics.stdev(timings,mean) if i>1 else 0))










