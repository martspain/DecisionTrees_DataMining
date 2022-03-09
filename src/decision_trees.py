import random
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sklearn
# import matplotlib.pyplot as plt
# from sklearn import datasets
# import seaborn as sb
# import sklearn.cluster as cluster
# import sklearn.metrics as metrics
# import sklearn.preprocessing
# import scipy.cluster.hierarchy as sch
from sklearn.model_selection import train_test_split
#from fcmeans import FCM
# import skfuzzy as fuzz
# import pylab
# import sklearn.mixture as mixture
import pyclustertend
from sklearn import cluster 
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
    def hopkins(self):
        df = self.df
        self.X = np.array(df[[
            'LotArea','OverallQual', 'TotRmsAbvGrd', 'GarageCars', 'FullBath'
        ]])
        X = self.X
        self.Y = np.array(df[['SalePrice']])
        random.seed(10000)
        X_scale = sklearn.preprocessing.scale(X)
        hop = pyclustertend.hopkins(X,len(X))

        return hop, X_scale, X

    # fuzzy c-means algorithms 
    def fuzzy_cMeans(self):
        hop, X_scale, X = self.hopkins()
        
        fcm = FCM(n_clusters = 5)
        fcm.fit(X)

        fcm_centers = fcm.centers
        fcm_labels = fcm.predict(X)

        plt.title("Grouping by Fuzzy C-Means")

        plt.scatter(X[:,0],X[:,1], c=fcm_labels, cmap='plasma')
        plt.scatter(fcm_centers[:,0],fcm_centers[:,1], c='green', marker='v')
        plt.show()
        
    def clusterNum(self):
        hop, X_scale, X = self.hopkins()
        numeroClusters = range(1,11)
        wcss = []
        for i in numeroClusters:
            kmeans = cluster.KMeans(n_clusters=i)
            kmeans.fit(X_scale)
            wcss.append(kmeans.inertia_)
        # plt.plot(numeroClusters, wcss)
        # plt.xlabel("Número de clusters")
        # plt.ylabel("Score")
        # plt.title("Gráfico de Codo")
        # plt.show()
    
    def percentile(self):
        x = self.df['SalePrice']
        threshold = x.quantile([0.33,0.67])
        self.firstRange, self.secondRange = threshold.iloc[0], threshold.iloc[1]

        return self.firstRange, self.secondRange
    
    def groupBy_ResponseVar(self):

        fR, sR = self.percentile()
        df = self.df.copy()

        df['SaleRange'] = df['SalePrice'].apply(
            lambda x: 'Low' if x <= fR 
            else ('Medium' if (x > fR and x <= sR) else 'High'))
        df_balance = df.copy()
        df = df.groupby('SaleRange').size()

        df_balance['SaleRange'] = df_balance['SaleRange'].astype('category')
        print(df_balance)
        return df

    def trainTest(self):
        df = self.df
        y = df.pop("SalePrice") #La variable respuesta
        X = df #El   resto de los datos
        random.seed(10000)
        X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.3,train_size=0.7)

    


driver = main('../train.csv')

# driver.exploreData()
#print(driver.hopkins()[0])
#driver.fuzzy_cMeans()
# print(driver.clusterNum())
driver.groupBy_ResponseVar()