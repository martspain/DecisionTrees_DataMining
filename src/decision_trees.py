import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import datasets
# import seaborn as sb
# import sklearn.cluster as cluster
# import sklearn.metrics as metrics
# import sklearn.preprocessing
# import scipy.cluster.hierarchy as sch
# # from fcmeans import FCM
# import skfuzzy as fuzz
# import pylab
# import sklearn.mixture as mixture
# import pyclustertend 
# import random
# import matplotlib.cm as cm
from reader import Reader

class main(object):
    def __init__(self, csvFilePath):
        # Universal Doc
        self.csvDoc = csvFilePath
        # Classes
        R = Reader(csvFilePath)
        self.df = R.data
    
    def exploreData(self):
        df = self.df
        print('-------------------------------------')
        for elem in df:
            space = ''
            if len(elem) < 15:
                count = 15 - len(elem)
            else:
                count = 0
            for n in range(count):
                space += ' '

            print('| ' + elem + space + '| ' + str(df.dtypes[elem]) + ' |')
            print('-------------------------------------')
        # print(self.df.dtypes)
    

driver = main('train.csv')

driver.exploreData()