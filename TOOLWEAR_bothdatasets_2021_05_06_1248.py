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
from sklearn.metrics import mean_squared_error
from mpl_toolkits import mplot3d


#ucitavanje dataseta
A_dataset = pd.read_csv('A.csv',encoding='latin1')
B_dataset = pd.read_csv('B.csv',encoding='latin1')

#stavljanje dataseta u array
dataA = A_dataset.values
dataB = B_dataset.values

# ispisivanje podataka iz svakog dataseta
#print (type(dataA))
#print (type(dataB))

#spajanje mjerenja u jedan dataset
data = np.vstack((dataA, dataB))    #spajanje mjerenja
#print (data)
X = data [:,1:4]
y = data[:,4]  
#print(X)
#print(y)
y_true = y
#print(X.shape, y.shape)

#odvajanje uzoraka za trening
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
#print (X_test)
# Genertsko programiranje   
#
# Mijenja se broj populacija kao parametar genetskog programiranja i gledaju se izlazni 
#   parametri - R2, RMSE,predikcija, funkcija predikcije
#
#
est_gp = SymbolicRegressor(population_size=5000,
                           generations=2, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)#,metric='rmse')

# pozivanje regresije
est_gp.fit(X_train, y_train)

#predikcija trosenja materijala
y_predict = est_gp.predict(X_test) #(X)
print ('Predikcija potrosnje alata(Genetic Programming):',y_predict)


#funckija predikcije
print('Funkcija predikcije:',est_gp._program)

#RMSE Score 
rmse_score_gp = mean_squared_error(y_test,y_predict,squared=False)  # squeard false daje RMSE
print ('RMSE Genetic Programming:',rmse_score_gp,'\n')
# R2 test kvalitete
print('R2 Genetic Programming:',est_gp.score(X_test,y_test))
score_gp = est_gp.score(X_test,y_test)
print('\n','\n','\n')

print('-----------Decision Tree Regressor------------')

#++++++++++++++ usporedba s drugim metodama ++++++++++++++++

###Decision Tree#####
est_tree = DecisionTreeRegressor(max_depth=5)
est_tree.fit(X_train, y_train)

#predikcija trosenja matrijala
y_tree = est_tree.predict(X_test)
print ('Predikcija trošenja alata(Decision Tree):',y_tree,)
score_tree = est_tree.score(X_test, y_test)
#R2
print ('R2 Decision Tree regressor:',score_tree)
rmse_score_tree = mean_squared_error(y_test,y_predict,squared=False)  # squeard false daje RMSE
print ('RMSE  Decision Tree regressor:',rmse_score_tree,'\n')


print ('\n','\n','\n')

print ('-----------Random Foresst Regressor------------')
####Random Foresst Regressor
est_rf = RandomForestRegressor(n_estimators=100,max_depth=5)
est_rf.fit(X_train, y_train)

y_rf = est_rf.predict(X_test)
print ('Predikcija trošenja alata(Decision Tree):',y_rf,)
score_rf = est_rf.score(X_test, y_test)

print ('R2 Random Tree Regressor:',score_rf,)

rmse_score_rf = mean_squared_error(y_test,y_predict,squared=False)  # squeard false daje RMSE
print ('RMSE  random forest:',rmse_score_rf,'\n')


print ('-----------------grafički prikaz rezultata-----------')
print (y_test)
print(y_predict)

# UZIMANJE PODATAKA IZ VRIJEDNOSTI KOJE SU SE KORISTILE IZ DATASETA
x1 = X_test[:,0] #DOC
y1 = X_test[:,1] #CS
z1 = X_test[:,2] #FR

# Creating figure
fig = plt.figure(figsize = (16, 9))
ax = plt.axes(projection ="3d")

# Add x, y gridlines
ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.3,
        alpha = 0.2)

# Creating color map
#my_cmap = plt.get_cmap('hsv')

# Creating plot
sctt = ax.scatter3D(x1, y1, z1,
                    alpha = 0.8,
                    c = y_test,
                    marker ='o')
 
plt.title("Prikaz podataka koje su koristene u treningu")
ax.set_xlabel('X-axis DOC', fontweight ='bold')
ax.set_ylabel('Y-axis CS', fontweight ='bold')
ax.set_zlabel('Z-axisFR', fontweight ='bold')
#fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
 

# show plot
plt.show()





## 2D graff vidi jel to kaj valja uopce

#   
#fig = plt.figure(figsize=(12, 10))
#for i, (y, score, title) in enumerate([(y_test, None, "Ground Truth"),
#                                       (y_predict, score_gp, "SymbolicRegressor"),
#                                       (y_tree, score_tree, "DecisionTreeRegressor"),
#                                       (y_rf, score_rf, "RandomForestRegressor")]):
#    ax = fig.add_subplot(2, 2, i+1)
#    points = ax.scatter(x, y_test, color='green', alpha=0.5)
#    test = ax.scatter(x,y_predict,color='red', alpha=0.5)
#    plt.title(title)
#plt.show()

