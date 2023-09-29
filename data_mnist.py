import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

## loading des datas

(X_train,Y_train),(X_test,Y_test)  =mnist.load_data()

## affiche un chiffre
'''
import matplotlib.pyplot as plt
plt.figure(figsize=(16, 8))
for i in range(1, 10):
    plt.subplot(4, 5, i)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(Y_train[i])
    plt.tight_layout()
plt.show()
print(max(X_train[1][8]))
'''
n_train=X_train.shape[0]
n_test=X_test.shape[0]

X_train=(np.reshape(X_train,(n_train,784))/255).T
#Y_train=np.reshape(Y_train,(Y_train.shape[0],1)) #pour mon reseau
#X_train=X_train.T#pour mon reseau
#Y_train=Y_train.T#pour mon reseau
Y_train=to_categorical(Y_train,num_classes=10).T #pour programme opti

X_test=np.float16(np.reshape(X_test,(n_test,784))/255).T
#Y_test=np.reshape(Y_test,(Y_test.shape[0],1)) #pour mon reseau
#X_test=X_test.T#pour mon reseau
#Y_test=Y_test.T#pour mon reseau
#Y_test=to_categorical(Y_test,num_classes=10).astype("float16").T #pour programme opti

## selection des 0 et 1 uniquement

'''X_train1=np.zeros((12665, 784))
Y_train1=np.zeros((12665,1))
i=0
for k in range (n_train):
    if Y_train[k]==1:
        X_train1[i]=X_train[k]
        Y_train1[i]=1
        i+=1
    elif Y_train[k]==0:
        X_train1[i]=X_train[k]
        i+=1

X_test1=np.zeros((2115, 784))
Y_test1=np.zeros((2115,1))
i=0
for k in range (n_test):
    if Y_test[k]==1:
        X_test1[i]=X_test[k]
        Y_test1[i]=1
        i+=1
    elif Y_test[k]==0:
        X_test1[i]=X_test[k]
        i+=1

X_train1=X_train1.T
X_test1=X_test1.T
Y_test1=Y_test1.T
Y_train1=Y_train1.T
'''

