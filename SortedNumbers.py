

import unittest
import datetime
import Genetic


class SortedNumbersTest(unittest.TestCase):

    def testSortTenNumbers(self):
        self.SortNumber(10)

    def SortNumber(self, totalNumber):
        geneSet=[i for i in range(100)]
        startTime=datetime.datetime.now()

        def fnDisplay(candidate):
            Display(candidate, startTime)

        def fnGetFitness(genes):
            return GetFitness(genes)

        optimalFitness=Fitness(totalNumber, 0)
        best=Genetic.getBest(fnGetFitness, totalNumber, optimalFitness, geneSet, fnDisplay)

        self.assertTrue(not optimalFitness>best.Fitness)

    def testBenchmark(self):
        Genetic.Benchmark.Run(lambda: self.SortNumber(40))

class Fitness:
    NumberInSequenceCount=None
    TotalGap=None

    def __init__(self,numberInSequenceCount,totalGap):
        self.NumberInSequenceCount=numberInSequenceCount
        self.TotalGap=totalGap

    def __gt__(self, other):
        if self.NumberInSequenceCount != other.NumberInSequenceCount:
            return self.NumberInSequenceCount> other.NumberInSequenceCount
        return self.TotalGap<other.TotalGap

    def __str__(self):
        return "{0} Sequential, {1} TotalGap".format(self.NumberInSequenceCount, self.TotalGap)


def GetFitness(genes):
    fitness=1
    gap=0
    for i in range(1,len(genes)):
        if genes[i]>genes[i-1]:
            fitness+=1
        else:
            gap+=genes[i-1]-genes[i]
    return Fitness(fitness,gap)


def Display(candidate,startTime):
    timeD=datetime.datetime.now()-startTime
    print('{0}=>\t{1}\t{2}'.format(
        ', '.join(map(str,candidate.Genes)),
        candidate.Fitness,
        str(timeD)
    ))


if __name__=='__main__':
    unittest.main()






