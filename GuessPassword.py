

import datetime
import Genetic
import unittest


class GuessPasswordTest(unittest.TestCase):
    geneSet = ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.<>?/!@#$%^&*()_+=-`~'

    def testHelloPassword(self):
        target='Hello World!'
        self.guessPassword(target)

    def testProgram(self):
        target = 'I am a python programmer'
        self.guessPassword(target)

    def guessPassword(self, target):
        startTime=datetime.datetime.now()

        def fnGetFitness(genes):
            return getFitness(genes, target)

        def fnDisplay(candidate):
            display(candidate, startTime)

        optimalFitness=len(target)
        best = Genetic.getBest(fnGetFitness, len(target), optimalFitness,
                               self.geneSet, fnDisplay)
        self.assertEqual( ''.join(best.Genes),target)

    def testBenchmark(self):
        Genetic.Benchmark.Run(self.testHelloPassword)


def getFitness(genes, target):
    return sum(1 for expected, actual in zip(target, genes) if expected==actual)

def display(candidate, startTime):
    timeD=datetime.datetime.now()-startTime
    print('{0}\t{1}\t{2}'.format(''.join(candidate.Genes), candidate.Fitness, str(timeD)))

# if __name__=='__main__':
#     testHelloPassword()

if __name__=='__main__':
    unittest.main()