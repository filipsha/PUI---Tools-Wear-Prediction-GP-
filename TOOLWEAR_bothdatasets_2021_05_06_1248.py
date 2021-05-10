# -*- coding: utf-8 -*-
"""
Created on Tue May  4 21:30:59 2021

@author: Filip Shahini
"""

#importiranje knjiznica
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sympy import sympify

 
#ucitavanje dataseta
A_dataset = pd.read_csv('A.csv',encoding='latin1')
B_dataset = pd.read_csv('B.csv',encoding='latin1')

#stavljanje dataseta u array
dataA = A_dataset.values
dataB = B_dataset.values

# ispisivanje podataka iz svakog dataseta
print (type(dataA))
print (type(dataB))

#spajanje mjerenja u jedan dataset
data = np.vstack((dataA, dataB))    #spajanje mjerenja
#print (data)
X = data [:,1:4]
y = data[:,4]  
#print(X)
#print(y)
y_true = y
print(X.shape, y.shape)

#odvajanje uzoraka za trening
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# Genertsko programiranje
est_gp = SymbolicRegressor(population_size=5000,
                           generations=20, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)#,metric='rmse')

# pozivanje regresije
est_gp.fit(X_train, y_train)
# R2 test kvalitete
print('R2:',est_gp.score(X_test,y_test))

## RMSE TEST kvalitete



#predikcija trosenja materijala ?? 


tool_wear_predictions = est_gp.predict(X_test) #(X)
print ('predikcija potrosnje materijala',tool_wear_predictions)

#funckija predikcije
print('funkcija predikcije',est_gp._program)

#next_e = sympify((est_gp._program)
#next_e

# usporedba s drugim metodama
#est_tree = DecisionTreeRegressor(max_depth=5)
#est_tree.fit(X_train, y_train)
#est_rf = RandomForestRegressor(n_estimators=100,max_depth=5)
#est_rf.fit(X_train, y_train)
#
#
#y_gp = est_gp.predict(X_test)
#score_gp = est_gp.score(X_test, y_test)
#
#y_tree = est_tree.predict(X_test)
#score_tree = est_tree.score(X_test, y_test)
#
#y_rf = est_rf.predict(X_test)
#score_rf = est_rf.score(X_test, y_test)
#
#print (score_gp)
#print (score_tree)
#print (score_rf)
#
#
#
#X=np.arange(0,len(X),1)
#
#fig = plt.figure(figsize=(12, 10))
#for i, (y, score, title) in enumerate([(y_true, None, "Ground Truth"),
#                                       (y_gp, score_gp, "SymbolicRegressor"),
#                                       (y_tree, score_tree, "DecisionTreeRegressor"),
#                                       (y_rf, score_rf, "RandomForestRegressor")]):
#    ax = fig.add_subplot(2, 2, i+1)
#    points = ax.scatter(X, y_true, color='green', alpha=0.5)
#    test = ax.scatter(X_test,y,color='red', alpha=0.5)
#    plt.title(title)
#plt.show()

