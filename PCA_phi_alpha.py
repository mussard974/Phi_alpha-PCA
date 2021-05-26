import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from numpy.linalg import inv
from scipy import linalg
from sklearn import preprocessing
import matplotlib.animation as animation
import math

class Pca_alpha(object):
    '''
    PCA in Generalized Convex Spaces
    
    Attributes
    ---------
        alpha : float

    methods
    --------
        phi_alpha()
        phi_alpha_matrix()
        project()
        eigenvalues()
        cta_var()
        cta()
        ctr()
        outliers()
        hotelling()
        grubbs()
        plot3D()
        plot2D()
        cercle_correlation
    '''
    
    def __init__(self, alpha):
        
        self.alpha = alpha

    def phi_alpha(self, a):
        '''
        Define the function phi_alpha^-1
        :param alpha (float)
        '''
        if self.alpha > 0:
            if a < 0:
                return -np.abs(a)**(1/self.alpha)
            elif a == 0:
                return 0
            else:
                return a**(1/self.alpha)
        elif self.alpha < 0 :
            if a < 0:
                return -np.abs(a)**(1/self.alpha)
            elif a > 0:
                return a**(1/self.alpha)
            else:
                return "infini" 
        else:
            raise ZeroDivisionError("It is not possible to divide by 0")
    
    
    def phi_alpha_matrix(self, x, alpha):
        '''
        Define the function phi_alpha^-1 for matrix
        :param x (document .xlsx)
        :param alpha (float != 0)
        :out B (array)
        '''
        n, k = x.shape
        B = np.zeros_like(x)
        for j in range(k):
            for i in range(n):
                if alpha >0:
                    if x[i,j] < 0:
                        B[i,j] = -np.abs(x[i,j])**(1/alpha)
                    elif x[i,j] == 0:
                        B[i,j] = 0
                    else:
                        B[i,j] = x[i,j]**(1/alpha)
                elif alpha <0:
                    if x[i,j] < 0:
                        B[i,j] = -np.abs(x[i,j])**(1/alpha)
                    elif x[i,j]>0:
                        B[i,j] = x[i,j]**(1/alpha)
                else: 
                    ZeroDivisionError("It is not possible to divide by 0")    
        return B
    
    def phi_alpha_matrix_version_2(self, x, alpha):
        '''
        Define the function phi_alpha
        :param x (document .xlsx)
        :param alpha (float != 0)
        :out B (array)
        '''
        n, k = x.shape
        B = np.zeros_like(x)
        for j in range(k):
            for i in range(n):
                if alpha >0:
                    if x[i,j] < 0:
                        B[i,j] = -np.abs(x[i,j])**(alpha)
                    elif x[i,j] == 0:
                        B[i,j] = 0
                    else:
                        B[i,j] = x[i,j]**(alpha)
                elif alpha <0:
                    if x[i,j] < 0:
                        B[i,j] = -np.abs(x[i,j])**1/alpha
                    elif x[i,j]>0:
                        B[i,j] = x[i,j]**1/alpha
                else: 
                    ZeroDivisionError("It is not possible to divide by 0")    
        return B
    
    
    def project(self, x, a):
        '''
        Calculate the matrix of the projection 'F'
        :param x (document .xlsx)
        :param a (float)
        :out F_alpha (array)
        '''
        n, k = x.shape
        Z = preprocessing.scale(x)
        R = (1/n) * (Z.T @ Z)
        _, vecp = linalg.eig(R)
        F = np.real(Z @ vecp)
        F_alpha = self.phi_alpha_matrix(F, a)
        return F_alpha

    def eigenvalues(self, x):
        '''
        Calculate the eigenvalues
        :param x (document .xlsx)
        :out A (array)
        '''
        n, k = x.shape
        Z = preprocessing.scale(x)
        R = (1/n) * (Z.T @ Z)
        valp, _ = linalg.eig(R)
        A = np.zeros((k,3))
        A[:,0] = np.real(valp.T)
        A = self.phi_alpha_matrix(A, self.alpha)
        A[:,2] = (np.cumsum(A[:,0]) / np.sum(A[:,0]))*100
        A[:,1] = (A[:,0] / A[:,0].sum())*100
        return np.round(A, 2)

    
    
    
    def absolute_contributions_variables(self, x):
        '''
        Calculate the absolute contribution of variables
        :param x (document .xlsx)
        :out cta_var1, cta_var (array)
        '''
        n, k = x.shape
        Z = preprocessing.scale(x)
        R = (1/n) * (Z.T @ Z)
        valp, P = linalg.eig(R)
        P_sorted=np.insert(P,0,np.real(valp),axis=0)
        P_sorted=P_sorted.T[P_sorted.T[:,0].argsort()[::-1]]
        P=np.delete(P_sorted.T, 0, 0)
        coordonnee_cercle_correlation = np.diag(np.real(valp)**(0.5)) @ P.T
        cta_var = self.phi_alpha_matrix(coordonnee_cercle_correlation, self.alpha)
        cta_var1 =np.c_[x.columns, ((cta_var.T)**2 / np.sum((cta_var.T)**2, axis = 0))*100] 
        return cta_var1, coordonnee_cercle_correlation
    
    
    def absolute_contributions_observations(self, x):
        '''
        Calculate the the absolute contribution of indviduals
        :param x (document .xlsx)
        :out cta (array)
        '''
        n, k = x.shape
        cta =np.c_[x.index,(self.project(x, self.alpha)**2 / np.sum(self.project(x, self.alpha)**2, axis = 0))*100]
        return cta

    def relative_contributions_observations(self, x):
        '''
        Calculate the CTR 
        :param x (document .xlsx)
        :out ctr (array)
        '''
        n, k = x.shape
        ctr = ((self.project(x, self.alpha).T)**2 / np.sum((self.project(x, self.alpha).T)**2, axis = 0))*100
        return np.round(ctr.T, 1)
    
    def hotelling(self, x, prob):
        '''
        Calculate the optimal alpha
        :param x (document .xlsx)
        :param prob (float)
        :out np.argmin(np.asarray(list0))/10 + 0.1 (array)
        '''
        n, k = x.shape
        list0 = []
        for a in np.arange(0.1, 10, 0.1):
            F = self.project(x, a)
            list1 = []
            list2 = []
            Hotelling1 = (n**2)*(n-1)/((n**2-1)*(n-1)) * (F[:,0])**2 / np.var(F[:,0])
            for i in range(n):
                if len(Hotelling1) == 0:
                    print("Please use the classical PCA with alpha = 1")
                if Hotelling1[i] >= ss.f.ppf(1-prob,dfn=1,dfd=n-1):
                    list1.append(i)
            list0.append(len(list1))
        #return list0
        return np.argmin(np.asarray(list0))/10 + 0.1
        
    
    def plot3D(self, x, a):
        '''
        Show the scatter graph in 3D of the individuals
        :param (document .xlsx)
        :param a (float)
        :out graph (plt)
        '''
        n, k = x.shape
        fig = plt.figure(1, figsize=(8, 6))
        ax = Axes3D(fig, elev=-150, azim=110)
       
        for line in range(n):
            ax.scatter(self.project(x, a)[line, 0], self.project(x, a)[line, 1], self.project(x, a)[line, 2], cmap=plt.cm.Set1, edgecolor='k', s=40)
            ax.text(self.project(x, a)[line, 0], self.project(x, a)[line, 1], self.project(x, a)[line, 2], s=x.index[line], size=10,  color='k') 
        ax.set_xlabel("1st component")
        ax.w_xaxis.set_ticklabels([])
        ax.set_ylabel("2nd component")
        ax.w_yaxis.set_ticklabels([])
        ax.set_zlabel("3rd component")
        ax.w_zaxis.set_ticklabels([])
        #def rotate(angle):
            #ax.view_init(azim=angle)
        #print("Making animation")
        #rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 2), interval=100)
        #rot_animation.save('BP.gif', dpi=80, writer='imagemagick')
    
    def plot2D(self, x, a):
        '''
        Show the scatter graph in 2D of the individuals
        :param x (document .xlsx)
        :param a (float)
        :out graph (plt)
        '''
        n, k = x.shape
        for line in range(n):
            plt.annotate(x.index[line],(self.project(x, a)[line,0],self.project(x, a)[line,1]))
        F_1=self.project(x, a)[:, 0]
        F_2=self.project(x, a)[:, 1]
        plt.scatter(F_1, F_2)
        plt.grid('on')
        plt.title('Point cloud graph')
        plt.xlabel('F_1')
        plt.ylabel('F_2')
        return plt.show()
    
    def circle_correlation(self, x):
        '''
        Show the circle of correlation of the variables 
        param: document (.xlsx)
        out: graph (plt)
        '''
        n, k = x.shape
        fig, axes = plt.subplots(figsize=(9,9))
        axes.set_xlim(-1,1)
        axes.set_ylim(-1,1)
        cercle = plt.Circle((0,0),1,color='blue',fill=False)
        axes.add_artist(cercle)
        vecteur_couleur=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        #Tracer les vecteurs
        for colonne in range(k):
            if k <=8:
                axes.quiver(np.array((0)), np.array((0)), self.absolute_contributions_variables(x)[1][0, colonne], self.absolute_contributions_variables(x)[1][1,colonne], color=vecteur_couleur[colonne], units='xy' ,scale=1, label=x.columns[colonne])
                plt.annotate(x.columns[colonne],(self.absolute_contributions_variables(x)[1][0,colonne],self.absolute_contributions_variables(x)[1][1,colonne]))
            else: 
                axes.quiver(np.array((0)), np.array((0)), self.absolute_contributions_variables(x)[1][0, colonne], self.absolute_contributions_variables(x)[1][1,colonne], color='k', units='xy' ,scale=1, label=x.columns[colonne])
                plt.annotate(x.columns[colonne],(self.absolute_contributions_variables(x)[1][0,colonne],self.absolute_contributions_variables(x)[1][1,colonne]))
        
        plt.legend()
        plt.xlabel('F_1')
        plt.ylabel('F_2')
        plt.grid(linestyle='--')
        plt.title('Cercle des correlations', fontsize=10)
        plt.show()
        