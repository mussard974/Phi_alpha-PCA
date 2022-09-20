##########################################################################################
######################## PCA in generalized convex spaces in Pytorch #####################
##########################################################################################

#Importation
import numpy as np
import pandas as pd
import torch
import scipy.stats as ss
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from numpy.linalg import inv
from numpy import genfromtxt
from scipy import linalg
import csv
import matplotlib.animation as animation
import math
import time
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from torchvision import datasets as dts
from torchvision.transforms import ToTensor 
from torchvision import transforms
from torchvision import datasets as dataset

class PCAAlpha_mean(torch.nn.Module):
    '''
    PCA in Generalized Convex Spaces
    
    Attributes
    ---------
        alpha : float
        n_comp : int number of components to keep

    methods
    --------
        phi_alpha()
        phi_alpha_matrix()
        eigenvalues()
        eigenvalues_2()
        project()
        project_2()
        plot3D()
    '''
    
    def __init__(self, alpha, nb_components=3):
        super().__init__()
        self.alpha = torch.tensor(alpha)
        self.nb_components = nb_components
    
    def phi_alpha(self, a, is_inverse=False):
        '''
        :param a (float) parameter alpha
        :param is_inverse (boolean) function phi_alpha or phi_alpha^-1 ?
        '''
        alpha = 1/self.alpha if is_inverse else self.alpha
        return torch.sgn(a) * (abs(a)**alpha)
    
    def phi_preprocess(self, x):
        '''
        :param x (document .xlsx)
        '''
        x_ = self.phi_alpha(x) 
        x1 = x_ - x_.mean(dim=0, keepdim=True)
        return self.phi_alpha(x1, is_inverse = True)

    def phi_preprocess_train_test(self, x_train, x_test):
        '''
        :param x (document .xlsx)
        '''
        x_phi = self.phi_alpha(x_train)
        x_ = self.phi_alpha(x_test) - x_phi.mean(dim=0, keepdim=True)
        self.mean_train_test = x_phi.mean(dim=0, keepdim=True)
        return self.phi_alpha(x_, is_inverse = True)


    def eigenvalues_2(self, x):
        '''
        Calculate the eigenvalues with centered points in the phi alpha space
        :param x (document .xlsx)
        :out A (array)
        '''
        n, k = x.shape
        Z = torch.tensor(self.phi_preprocess(x))
        R = (1/n) * self.phi_alpha(Z.T) @ self.phi_alpha(Z)
        valp = torch.real(torch.linalg.eigvals(R))
        A = torch.zeros([k,3]) 
        A[:,0] = self.phi_alpha(valp.T, is_inverse=True) 
        A[:,2] = (torch.cumsum(A[:,0], dim=0) / torch.sum(A[:,0]))*100
        A[:,1] = (A[:,0] / A[:,0].sum())*100
        return print("valeurs propres 2", torch.real(A))
    
    def fit_2(self, x):
        '''
        Calculate the matrix of projection 'F' with centered points in the phi alpha space
        :param x (document .xlsx)
        :nbre_comp (int): nbre de composantes principales à extraire
        :out F_alpha (array)
        '''
        n, k = x.shape
        z = self.phi_preprocess(x)
        z = self.phi_alpha(z)
        valp, vecp = torch.linalg.eig(z.T @ z /n)
        F = z @ torch.real(vecp)
        F = self.phi_alpha(F, is_inverse=True)
        return F, valp, torch.real(vecp)

    def fit_inverse(self, x_train, x_test):
        '''
        Calculate the projection onto the space of Z
        '''
        z = self.phi_preprocess_train_test(x_train, x_test)
        vecp = self.fit_2(x_train)[2]
        F = self.phi_alpha(z) @ vecp
        F[:, self.nb_components::] = 0
        F_inverse = F @ torch.inverse(vecp)
        F1 = F_inverse + self.mean_train_test
        return self.phi_alpha(F1, is_inverse=True)
    
    
    def absolute_contributions_variables(self, x):
        '''
        Calculate the absolute contribution of variables
        :param x (document .xlsx)
        :out cta_var1, cta_var (array)
        '''
        #n, k = x.shape
        #Z = torch.tensor(self.phi_preprocess(x))
        #R = (1/n) * self.phi_alpha(Z.T) @ self.phi_alpha(Z)
        #valp = torch.real(torch.linalg.eigvals(R))
        #_, P = torch.linalg.eig(R)
        #P = torch.real(P)
        #valp2 = torch.reshape(valp, (1, 784))
        #P_sorted = torch.cat([P[:0], valp2, P[0:]], 0)
        #P_sorted_trie=torch.argsort(P_sorted.T[:,0], dim=-1)
        #P_sorted_trie = torch.flip(P_sorted_trie, dims=(0,))
        #P_sorted = P_sorted.T[P_sorted_trie]
        #coordonnee_cercle_correlation = torch.diag(torch.real(valp)**(0.5)) @ P.T
        #cta_var = self.phi_alpha(coordonnee_cercle_correlation)
        #cta_var1 =np.c_[x.columns, ((cta_var.T)**2 / np.sum((cta_var.T)**2, axis = 0))*100] 
        #cta_var1 = torch.stack(x.columns,((cta_var.T)**2) / np.sum((cta_var.T)**2),dim=1)*100
        #return cta_var1, coordonnee_cercle_correlation
        n, k = x.shape
        F, _ = self.fit_2(x)
        cta = (torch.pow(F,2) / torch.sum(torch.pow(F,2), axis = 0))*100
        return cta
    
    def hotelling(self, x, prob):
        "optimal alpha"
        n, k = x.shape
        list0 = []
        for a in torch.arange(0.1, 10, 0.1):
            F, _ = self.fit_2(x)
            list1 = []
            list2 = []
            Hotelling1 = (n**2)*(n-1)/((n**2-1)*(n-1)) * (F[:,0])**2 / torch.var(F[:,0], unbiased=False)
            for i in range(n):
                if Hotelling1[i] >= ss.f.ppf(1-prob,dfn=1,dfd=n-1):
                    list1.append(i)
            list0.append(len(list1))
        #return list0
        return torch.argmin(torch.asarray(list0))/10 + 0.1

    def plot3D2(self, x, header = False):
        '''
        Show the scatter graph in 3D of the individuals with centered points in the phi alpha space
        :param (document .xlsx)
        :param a (float)
        :out graph (plt)
        '''
        _, fit  = self.fit_2(x)
        n, k = x.shape
        fig = plt.figure(2, figsize=(8, 6))
        ax = Axes3D(fig, elev=-150, azim=110)
       
        for line in range(n):
            ax.scatter(self.fit_2[line, 0], self.fit_2[line, 1], self.fit_2[line, 2], cmap=plt.cm.Set1, edgecolor='k', s=40)
            if(header):
              ax.text(self.fit_2[line, 0], self.fit_2[line, 1], self.fit_2[line, 2], s=individus[line], size=10,  color='k') 
        ax.set_xlabel("1st component")
        ax.w_xaxis.set_ticklabels([])
        ax.set_ylabel("2nd component")
        ax.w_yaxis.set_ticklabels([])
        ax.set_zlabel("3rd component")
        ax.w_zaxis.set_ticklabels([])
        plt.title("Plot 3D of the individuals with centered points in the phi alpha space")
        #def rotate(angle):
            #ax.view_init(azim=angle)
        #print("Making animation")
        #rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 2), interval=100)
        #rot_animation.save('rotation.gif', dpi=80, writer='imagemagick')

