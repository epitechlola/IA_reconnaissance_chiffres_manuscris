from reseau_n_couches import *
from sklearn.metrics import accuracy_score

def test (X_t,y_t,parametres):
    y_pred=predict(X_t,parametres)#prediction des resultats sur donnees test
    accu=accuracy_score(y_t.flatten(),y_pred.flatten())#check des predictions / resultat vrai
    return accu,y_pred