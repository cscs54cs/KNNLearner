import math,random,sys,bisect,time
import numpy,scipy.spatial.distance
from scipy.spatial import cKDTree
import cProfile,pstats
import numpy as np
import csv as csv
import string
class KNNLearner:
    
    def __init__(self,k=3):
        self.k = k
        self.data = None
    def addEvidence(self, dataX, dataY=None):
        if not dataY == None:
            data = numpy.zeros([dataX.shape[0],dataX.shape[1]+1])
            data[:,0:dataX.shape[1]]=dataX
            data[:,(dataX.shape[1])]=dataY
        else:
            data = dataX

        if self.data is None:
            self.data = data
        else:
            self.data = numpy.append(self.data,data,axis=0)
            
    def query(self,point,k=None):
        if k is None:
            k = self.k
        train=numpy.zeros([self.data.shape[0],self.data.shape[1]+1])
        train[:,1:self.data.shape[1]+1]=self.data
        for tp in train:
            tp[0]=numpy.sqrt(numpy.square(point[0]-tp[1])+numpy.square(point[1]-tp[2]) )
        ide=numpy.argsort(train,axis =0)
        i=0
        expect = 0
        for i in range(0,k):
            expect= expect+train[ide[i][0],3]
        expect=expect/k
        return expect



def readdata(filename1):
    reader= csv.reader(open(filename1,'rU'),delimiter=',')
    learner = KNNLearner(3)
    for row in reader:
        temp = numpy.zeros([1,3])
        i=0
        for elements in row:
            temp[0][i]=string.atof(elements)
            i=i+1
        learner.addEvidence(temp)
    learner.query([0,0])
    return learner.data
def test():
    datat = readdata('data-classification-prob.csv')
    print datat
if __name__=="__main__":
    test()
