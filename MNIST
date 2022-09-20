### Application on MNIST

!git clone https://github.com/EscVM/Efficient-CapsNet.git

# move the working directory to Efficient-CapsNet folder
import os
os.chdir('Efficient-CapsNet')

!pip install -r requirements.txt


############ 
# MNIST
############

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

B = np.zeros((30, 1))
C = np.zeros((30, 1))
D = np.zeros((30, 1))
E = np.zeros((30, 1))

i = 0
for nb_composantes in np.arange(10, 110, 10):
    SVD = TruncatedSVD(n_components=nb_composantes, n_iter=5, random_state=42)
    transformer = KernelPCA(n_components=nb_composantes, kernel='linear', fit_inverse_transform=True)
    pca = PCA(n_components=nb_composantes)
    sparse = SparsePCA(n_components=nb_composantes, random_state=0)

    X_test_PCA = pca.fit_transform(x_test)
    X_test_PCA = pca.inverse_transform(X_test_PCA)
    X_test_PCA  = torch.reshape(torch.from_numpy(X_test_PCA), (10000, 28, 28,1))
    X_test_PCA = X_test_PCA.numpy()
    resultat_PCA = model_test.evaluate(X_test_PCA,y_test)
    D[i, 0] = resultat_PCA


    X_transformed = transformer.fit_transform(x_test)
    X_test_KernelPCA = transformer.inverse_transform(X_transformed)
    X_test_KernelPCA  = torch.reshape(torch.from_numpy(X_test_KernelPCA), (10000, 28, 28,1))
    X_test_KernelPCA = X_test_KernelPCA.numpy()
    resultat_KernelPCA = model_test.evaluate(X_test_KernelPCA,y_test)
    C[i, 0] = resultat_KernelPCA



    X_test_SVD = SVD.fit_transform(x_test)
    X_test_SVD = SVD.inverse_transform(X_test_SVD)
    X_test_SVD  = torch.reshape(torch.from_numpy(X_test_SVD), (10000, 28, 28,1))
    X_test_SVD = X_test_SVD.numpy()
    resultat = model_test.evaluate(X_test_SVD,y_test)
    B[i, 0] = resultat
    
    
    X_test_sparse = sparse.fit_transform(x_test)
    X_test_sparse = (X_test_sparse @ sparse.components_) + sparse.mean_
    X_test_sparse  = torch.reshape(torch.from_numpy(X_test_sparse), (10000, 28, 28,1))
    X_test_sparse = X_test_sparse.numpy()
    resultat = model_test.evaluate(X_test_sparse,y_test)
    E[i, 0] = resultat

    i+=1 
