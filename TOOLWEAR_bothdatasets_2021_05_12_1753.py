# -*- coding: utf-8 -*-
"""
Created on Tue May  4 21:30:59 2021

@author: Filip Shahini
"""
#%%
#importiranje knjiznica
import numpy as np
import pandas as pd
#from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from my import GP_Train

#ucitavanje dataseta
A_dataset = pd.read_csv('A.csv',encoding='latin1')
B_dataset = pd.read_csv('B.csv',encoding='latin1')

#stavljanje dataseta u array
dataA = A_dataset.values
dataB = B_dataset.values
generations = [1, 5, 10, 20, 30, 50, 70, 100]

# ispisivanje podataka iz svakog dataseta
#print (type(dataA))
#print (type(dataB))
print ('Predefined generations which are going to be used for training are',generations)
answer = str(input('What dataset do you want to train (A,B or A+B)?'))
#%%
######++++++++++++Dataset A+++++++++++++######
if answer == 'A':
    print ('Dataset A selected. GP will start shortly')
    X = dataA [:,1:4]
    y = dataA[:,4]
    y_true = y

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    num_gen = len(generations)
    num_of_test_samples = len(X_test)
    #r2 = []*100
    #rmse = [] *100
    r2 = np.empty(num_gen)
    rmse = np.empty(num_gen, dtype=object)
    print('duzima',len(r2))
    print  ('uzorci za trening:',y_test)
    for i in range (0,len(generations),1):
    
        out = GP_Train(X,y,generations[i])
        r2[i] = round(out[0],3)
        rmse[i] = round(out [1],3)

#%%
        ## dodati vrijednosti pored R2 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot (generations,r2,'-o')
    plt.grid()
    plt.title ('R2 - Generation Size Corelation')
    plt.ylabel  ('R2')
    plt.xlabel ('Number of generations')

    plt.show()
    
    plt.plot (generations,rmse,'-o')
    plt.title ('RMSE - Generation Size Corelation')
    plt.ylabel  ('RMSE')
    plt.xlabel ('Number of generations')
    plt.grid()
    plt.show()
    print ('This are all R2 values',r2)
    print ('This are all RMSE values',rmse)
    
    #++++++++++++Dataset B+++++++++++++++++++++++++#
#%%
elif answer == 'B': 
    print ('Dataset A selected. GP will start shortly')    
    X = dataB [:,1:4]
    y = dataB[:,4]
    y_true = y

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    num_gen = len(generations)

    #r2 = []*100
    #rmse = [] *100
    r2 = np.empty(num_gen, dtype=object)
    rmse = np.empty(num_gen, dtype=object)
    print('duzina',len(r2))
    for i in range (0,len(generations),1):
    
        out = GP_Train(X,y,generations[i])
        r2[i] = out[0]
        rmse[i] = out [1]
#%%
    plt.plot (generations,r2,'-o')
    plt.title ('R2 - Generation Size Corelation')
    plt.ylabel  ('R2')
    plt.xlabel ('Number of generations')
    plt.grid()
    plt.show()
    
    plt.plot (generations,rmse,'-o')
    plt.title ('RMSE - Generation Size Corelation')
    plt.ylabel  ('RMSE')
    plt.xlabel ('Number of generations')
    plt.grid()
    plt.show()
    

    
    #+++++++++++++obadataseta
#%%
elif answer == 'A+B':
    print ('Datasets A+B selected. GP will start shortly')
    data = np.vstack((dataA, dataB))    #spajanje mjerenja
    #print (type(data))
    X = data [:,1:4]
    y = data[:,4]  
    #print(X)
    #print(y)
    y_true = y
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    #print(X.shape, y.shape)
    
    num_gen = len(generations)
    
    #r2 = []*100
    #rmse = [] *100
    r2 = np.empty(num_gen, dtype=object)
    rmse = np.empty(num_gen, dtype=object)
    print('duzima',len(r2))
    for i in range (0,len(generations),1):
    
        out = GP_Train(X,y,generations[i])
        r2[i] = out[0]
        rmse[i] = out [1]
        
    plt.plot (generations,r2,'-o')
    plt.title ('R2 - Generation Size Corelation')
    plt.ylabel  ('R2')
    plt.xlabel ('Number of generations')
    plt.grid()
    plt.show()
    
    plt.plot (generations,rmse,'-o')
    plt.title ('RMSE - Generation Size Corelation')
    plt.ylabel  ('RMSE')
    plt.xlabel ('Number of generations')
    plt.grid()
    plt.show()
    
 
#%%
#answer = str(input('Do you want to try other methods (Yes/No)?'))
#if answer == 'Yes':
print('\n','\n','\n')
print('-----------Decision Tree Regressor------------')


#++++++++++++++ usporedba s drugim metodama ++++++++++++++++

##Decision Tree#####
est_tree = DecisionTreeRegressor(max_depth=5)
est_tree.fit(X_train, y_train)

#predikcija trosenja matrijala
y_tree = est_tree.predict(X_test)
print ('Predikcija trošenja alata(Decision Tree):',y_tree,)
score_tree = est_tree.score(X_test, y_test)
#R2
print ('R2 Decision Tree regressor:',score_tree)
rmse_score_tree = mean_squared_error(y_test,y_tree,squared=False)  # squeard false daje RMSE
print ('RMSE  Decision Tree regressor:',rmse_score_tree,'\n')


print ('\n','\n','\n')

print ('-----------Random Foresst Regressor------------')
####Random Foresst Regressor
est_rf = RandomForestRegressor(n_estimators=100,max_depth=5)
est_rf.fit(X_train, y_train)

y_rf = est_rf.predict(X_test)
print ('Predikcija trošenja alata(Random Foresst):',y_rf,)
score_rf = est_rf.score(X_test, y_test)

print ('R2 Random Tree Regressor:',score_rf,)

rmse_score_rf = mean_squared_error(y_test,y_rf,squared=False)  # squeard false daje RMSE
print ('RMSE  random forest:',rmse_score_rf,'\n')



print ('-----------------grafički prikaz rezultata-----------')
print ('Testni uzorci:',y_test)

#print(y_predict)
y_predict = out[2]
print ('Predikted values:',y_predict)
# UZIMANJE PODATAKA IZ VRIJEDNOSTI KOJE SU SE KORISTILE IZ DATASETA
x1 = X_test[:,0]  #DOC
y1 = X_test[:,1] #CS
z1 = X_test[:,2] #FR
c = y_test
# Creating figure
fig = plt.figure(figsize = (16, 9))
ax = fig.add_subplot(111, projection='3d')

# Add x, y gridlines
ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.3,
        alpha = 0.2)

# Creating color map
#my_cmap = plt.get_cmap('hsv')

# Creating plot
 
sctt = ax.scatter(x1, y1, z1,
                    c = y_test,
                    cmap='winter_r',
                    marker ='o',
                    s=300,
                    label = 'test'
                    )
sctt1 = ax.scatter3D(x1, y1, z1,
                    c = y_predict,
                    cmap='winter_r',
                    marker ='x',
                    s=300,
                    label = 'predicted'
                    )
fig.colorbar(sctt,label='Test samples')#,boundaries=np.linspace(0, 160, 10))
fig.colorbar(sctt1, label = "Predicted values")#,boundaries=np.linspace(0, 160, 20))
plt.legend()
plt.title("Test vs predicted values in dependence of dataset input parameters")
ax.set_xlabel('X-axis DOC', fontweight ='bold')
ax.set_ylabel('Y-axis CS', fontweight ='bold')
ax.set_zlabel('Z-axis FR', fontweight ='bold')

#ax.text(x1, y1, z1,' %s' (y_predict.value), size=20, zorder=1)
#fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
 

# show plot
plt.show()
#



  


# show plot

#
#
#
#
#
## 2D graff vidi jel to kaj valja uopce
#
#   
#fig = plt.figure(figsize=(12, 10))
#for i, (y, score, title) in enumerate([(y_test, None, "Ground Truth"),
#                                       (y_predict, score_gp, "SymbolicRegressor"),
#                                       (y_tree, score_tree, "DecisionTreeRegressor"),
#                                       (y_rf, score_rf, "RandomForestRegressor")]):
#    ax = fig.add_subplot(2, 2, i+1)
#    points = ax.scatter(x, y_test, color='green', alpha=0.5)
#    test = ax.scatter(x,y_predict,color='red', alpha=0.5)
#    plt.title(title)A
#plt.show()

