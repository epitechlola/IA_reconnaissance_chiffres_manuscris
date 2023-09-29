import numpy as np

def initialisation (taille_reseau):
    parametres={}
    for k in range (1,len(taille_reseau)):
        parametres['W'+str(k)]=np.random.randn(taille_reseau[k],taille_reseau[k-1])
        parametres['b'+str(k)]=np.random.randn(taille_reseau[k],1)

    return parametres

def forward_propagation(X,parametres):
    activations={}

    activations['A0']=X

    for k in range (1,(len(parametres)//2)+1):
        Z=parametres['W'+str(k)].dot(activations['A'+str(k-1)])+parametres['b'+str(k)]
        activations['A'+str(k)]=1/(1+np.exp(Z))

    return activations

def back_propagation(activations,X,y,parametres):
    gradients={}

    m=y.shape[1]

    dZ=activations['A'+str(len(parametres)//2)]-y

    for k in reversed (range (1,len(parametres)//2+1)):

        gradients['dW'+str(k)]=1/m *dZ.dot(activations['A'+str(k-1)].T)
        gradients['db'+str(k)]=1/m *np.sum(dZ,axis=1,keepdims=True)
        dZ=np.dot(parametres['W'+str(k)].T,dZ)*activations['A'+str(k-1)]*(1-activations['A'+str(k-1)])

    return gradients

def update (gradients,parametres,alpha):
    for k in range (1,len(parametres)//2+1):
        parametres['W'+str(k)]=parametres['W'+str(k)]-alpha*gradients['dW'+str(k)]
        parametres['b'+str(k)]=parametres['b'+str(k)]-alpha*gradients['db'+str(k)]

    return parametres

#pour reponse binaire 1/0
"""def predict (X, parametres):
    activations=forward_propagation(X,parametres)
    A2=activations['A2']

    return A2>=0.5"""

#pour reponse avec choix > 2
def predict (X,parametres):
    activations=forward_propagation(X,parametres)
    A=activations['A'+str(len(parametres)//2)].T
    prediction=np.zeros(len(A))

    for k in range (len(A)):
        prediction[k]=(np.argmax(A[k]))

    return prediction

def reseau (X,y,dimension,alpha,iter):

    taille_reseau=dimension
    taille_reseau.append(y.shape[0])
    taille_reseau.insert(0,X.shape[0])

    np.random.seed(0)
    parametres=initialisation(taille_reseau)

    for i in range (iter):
        activations=forward_propagation(X,parametres)
        gradients=back_propagation(activations,X,y,parametres)
        parametres=update(gradients,parametres,alpha)

    return parametres












