import math,random,sys,bisect,time
import numpy,scipy.spatial.distance
from scipy.spatial import cKDTree
import cProfile,pstats
import numpy as np
import csv as csv
import string
import matplotlib.pyplot as plt
class LinRegLearner:
    
    def __init__(self):
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
            
    def query(self,point):
        expect = point[0]*self.w1[0]+point[1]*self.w1[1]+self.w1[2]
        return expect
    
    def ana(self):
        X = numpy.ones([self.data.shape[0],self.data.shape[1]])
        Y = numpy.zeros([self.data.shape[0],1])
        X[:,0:self.data.shape[1]-1] = self.data[:,0:self.data.shape[1]-1]
        Y = self.data[:,self.data.shape[1]-1]
        x1 = np.array(X)
        self.w1 = numpy.linalg.lstsq(X,Y)[0]

def readdata(filename1):
    reader= csv.reader(open(filename1,'rU'),delimiter=',')
    learner = LinRegLearner()
    for row in reader:
        temp = numpy.zeros([1,3])
        i=0
        for elements in row:
            temp[0][i]=string.atof(elements)
            i=i+1
        learner.addEvidence(temp)
    learner.ana()
    for row in learner.data:
        ye = learner.query(row[0:learner.data.shape[1]-1])
        print ye , row[learner.data.shape[1]-1]
    return learner.data
def test():
    datat = readdata('data-classification-prob.csv')
if __name__=="__main__":
    test()
