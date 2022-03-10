from locale import normalize
import random
from re import X
from tkinter import Y
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn import tree
from sklearn.model_selection import train_test_split
from fcmeans import FCM
import pyclustertend
from sklearn import cluster 
from reader import Reader




class main(object):
    def __init__(self, csvFilePath):
        # Universal Doc
        self.csvDoc = csvFilePath
        # Classes
        R = Reader(csvFilePath)
        self.df = R.data
        

    def shape(self):
        df = self.df

        print(df.shape)
        print(df.head())
        print(df.isnull().sum())

    def hopkins(self):
        df = self.df
        column_names = ['LotArea','OverallQual', 'TotRmsAbvGrd', 'GarageCars', 'FullBath']
        row_names = ['SalePrice']
        self.X = np.array(df[column_names])
        X = self.X
        self.Y = np.array(df[row_names])
        
        Y = self.Y
        random.seed(10000)
        X_scale = sklearn.preprocessing.scale(X)
        hop = pyclustertend.hopkins(X,len(X))
        
        df = pd.DataFrame(df, columns=column_names)
        return hop, X_scale, X, Y, df

    # fuzzy c-means algorithms 
    def fuzzy_cMeans(self):
        hop, X_scale, X, Y, df = self.hopkins()
        
        fcm = FCM(n_clusters = 5)
        fcm.fit(X)

        fcm_centers = fcm.centers
        fcm_labels = fcm.predict(X)

        plt.title("Grouping by Fuzzy C-Means")

        plt.scatter(X[:,0],X[:,1], c=fcm_labels, cmap='plasma')
        plt.scatter(fcm_centers[:,0],fcm_centers[:,1], c='green', marker='v')
        plt.show()
        
    def clusterNum(self):
        hop, X_scale, X, Y, df = self.hopkins()
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
        # print("\n df")
        return df
        
    def trainTest(self):
        df = self.df
        hop, X_scale, X, Y, df = self.hopkins()
  

        X_train, X_test,y_train, y_test = train_test_split(X, Y, random_state =101, test_size=0.3,train_size=0.7)
      
        return X_train, X_test,y_train, y_test

    def treeDepth(self):
        df = self.df
        X_train, X_test,y_train, y_test = self.trainTest()

        train_accuracy = []
        test_accuracy = []

        for depth in range(1, 10):
            df = tree.DecisionTreeClassifier(max_depth=depth, random_state=10)
            df.fit(X_train, y_train)
            
            train_accuracy.append(df.score(X_train, y_train))
            test_accuracy.append(df.score(X_test, y_test))

        frame = pd.DataFrame({'max_depth':range(1, 10), 'train_acc':train_accuracy, 'test_acc':test_accuracy})
        print(frame.head())

        # plt.figure(figsize=(12, 6))
        # plt.plot(frame['max_depth'], frame['train_acc'], marker='o')
        # plt.plot(frame['max_depth'], frame['test_acc'], marker='o')
        # plt.xlabel('Depth of tree')
        # plt.ylabel('Performance')
        # plt.legend()
        # plt.show()


        # EL DEPTH ES DE 3

    def decision_tree(self):
        X_train, X_test,y_train, y_test = self.trainTest()
        hop, X_scale, X, Y, df = self.hopkins()

        dt = tree.DecisionTreeClassifier(max_depth=3, random_state=10)
        dt.fit(X_train, y_train)

        feature_names = df.columns
        tree.export_graphviz(dt, out_file='tree.dot', feature_names=feature_names, class_names=True, max_depth=2)
        
        # para correrlo tiene que descargar graphviz
        # despues -> dot -Tpng tree.dot -o tree.png
    

driver = main('../train.csv')

# driver.exploreData()
#print(driver.hopkins()[0])
#driver.fuzzy_cMeans()
# print(driver.clusterNum())
driver.decision_tree()
    