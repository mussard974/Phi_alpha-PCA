### Application on MNIST

!git clone https://github.com/EscVM/Efficient-CapsNet.git

# move the working directory to Efficient-CapsNet folder
import os
os.chdir('Efficient-CapsNet')

!pip install -r requirements.txt


############ 
# MNIST
############
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import AffineVisualizer, Dataset
from models import EfficientCapsNet
mnist_dataset = Dataset('MNIST', config_path='config.json') # only MNIST

model_test = EfficientCapsNet('MNIST', mode='test', verbose=False)
model_test.load_graph_weights()
model_play = EfficientCapsNet('MNIST', mode='play', verbose=False)
model_play.load_graph_weights()

x_test = mnist_dataset.X_test
x_test = np.reshape(x_test, (10000, 784))
x_train = mnist_dataset.X_train
x_train = np.reshape(x_train, (60000, 784))
y_test = mnist_dataset.y_test
y_train = mnist_dataset.y_train

### Different PCA techniques ####
#SVD Kernel PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
        
# Matrices des résultats de la classif        
A = np.zeros((10, 1))
B = np.zeros((10, 1))
C = np.zeros((10, 1))
D = np.zeros((10, 1))
E = np.zeros((10, 1))

alpha_param = 0.4

i = 0
for nb_composantes in np.arange(10, 110, 10):
    model_Phi_Alpha_PCA = PCAAlpha(alpha = alpha_param, nb_components = nb_composantes)
    X_test = model_Phi_Alpha_PCA.fit_inverse(x_train, x_test)
    X_test  = torch.reshape(X_test, (10000, 28, 28,1))
    X_test = X_test.numpy()
    A[i, 0] = model_test.evaluate(X_test, y_test)
    
    pca = PCA(n_components=nb_composantes)
    pca.fit(x_train)
    X_test_PCA = pca.fit_transform(x_test)
    X_test_PCA = pca.inverse_transform(X_test_PCA)
    X_test_PCA  = torch.reshape(torch.from_numpy(X_test_PCA), (10000, 28, 28,1))
    X_test_PCA = X_test_PCA.numpy()
    B[i, 0] = model_test.evaluate(X_test_PCA,y_test)

    transformer = KernelPCA(n_components=nb_composantes, kernel='poly', fit_inverse_transform=True)
    transformer.fit(x_train)
    X_test_KernelPCA = transformer.fit_transform(x_test)
    X_test_KernelPCA = transformer.inverse_transform(X_test_KernelPCA)
    X_test_KernelPCA  = torch.reshape(torch.from_numpy(X_test_KernelPCA), (10000, 28, 28,1))
    X_test_KernelPCA = X_test_KernelPCA.numpy()
    C[i, 0] = model_test.evaluate(X_test_KernelPCA,y_test)
    
    SVD = TruncatedSVD(n_components=nb_composantes, n_iter=5, random_state=42)
    SVD.fit(x_train)
    X_test_SVD = SVD.fit_transform(x_test)
    X_test_SVD = SVD.inverse_transform(X_test_SVD)
    X_test_SVD  = torch.reshape(torch.from_numpy(X_test_SVD), (10000, 28, 28,1))
    X_test_SVD = X_test_SVD.numpy()
    D[i, 0] = model_test.evaluate(X_test_SVD,y_test)
    
    sparse = SparsePCA(n_components=nb_composantes, random_state=0)    
    sparse.fit(x_train)
    X_test_sparse = sparse.fit_transform(x_test)
    X_test_sparse = (X_test_sparse @ sparse.components_) + sparse.mean_
    X_test_sparse  = torch.reshape(torch.from_numpy(X_test_sparse), (10000, 28, 28,1))
    X_test_sparse = X_test_sparse.numpy()
    E[i, 0] = model_test.evaluate(X_test_sparse,y_test)

    i+=1 

 ### GRAPH
nb_comp = np.arange(10, 110, 10)
plt.plot(nb_comp, A[0:10, :], label= "alpha = 0.3")
plt.plot(nb_comp, B[0:10, :], label ="PCA")
plt.plot(nb_comp, C[0:10, :], label ="Kernel")
plt.plot(nb_comp, D[0:10, :], label ="SVD")
plt.plot(nb_comp, E[0:10, :], label ="Sparse")

plt.xlim(10,110)
plt.ylim(0,1)
plt.legend(loc="best")
plt.xlabel('Number of principal components')
plt.ylabel('Accuracy')
plt.title('Accuracy according to the principal components and the method used')
plt.show()
plt.savefig("graph_accuracy.png")
 
