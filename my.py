# -*- coding: utf-8 -*-
"""
Created on Tue May 11 21:44:32 2021

@author: Eris
"""

#importiranje knjiznica

from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def GP_Train (X,y,Gen):    #mijenja se samo broj generacija
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    #print (X_test)
    # Genertsko programiranje   
    #
    # Mijenja se broj populacija kao parametar genetskog programiranja i gledaju se izlazni 
    #   parametri - R2, RMSE,predikcija, funkcija predikcije
    #
    #
    
    est_gp = SymbolicRegressor(population_size=5000,
                               generations= Gen, stopping_criteria=0.01,
                               p_crossover=0.7, p_subtree_mutation=0.1,
                               p_hoist_mutation=0.05, p_point_mutation=0.1,
                               max_samples=0.9, verbose=1,
                               parsimony_coefficient=0.01, random_state=0)#,metric='rmse')
    
    # pozivanje regresije
    est_gp.fit(X_train, y_train)
    
    #predikcija trosenja materijala
    y_predict = est_gp.predict(X_test) #(X)
    print ('Predikcija potrosnje alata(Genetic Programming):',y_predict,'\n')
    
    
    #funckija predikcije
    print('Funkcija predikcije:',est_gp._program,'\n')
    
    
    # R2 test kvalitete
    print('R2 Genetic Programming:',est_gp.score(X_test,y_test))
    r2_score_gp = est_gp.score(X_test,y_test)
    #RMSE Score 
    rmse_score_gp = mean_squared_error(y_test,y_predict,squared=False)  # squeard false daje RMSE
    print ('RMSE Genetic Programming:',rmse_score_gp)



    return (r2_score_gp,rmse_score_gp,y_predict)