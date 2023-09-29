from time import time
from data_mnist import *
from reseau_n_couches import *
from test_reseau import *

t1=time()
param=reseau(X_train,Y_train,[8,8],0.20,50)
t2=time()
a=test(X_test,Y_test,param)
t3=time()
print("l'accuracy est de : ",a[0], " le modèle s'est entrainé pendant : ", t2-t1, " secondes et à mis  : ",t3-t2, " secondes a effectuer les tests.")