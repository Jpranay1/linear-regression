import numpy as np
import random

def gen(var,n,m):
        beta = np.random.randint(9,size=m+1)
        X = np.random.rand(n,m+1)
        X[:,0] = 1
        h = np.dot(X, beta)
        e = np.random.normal(0,var,n)
        Y = h #+ e
        X = np.delete(X,0,1)
        print("this is known beta")
        print(beta)
        return beta, X, Y

def costf(x,y,beta,m,n):
        h = np.dot(x, beta)
        r = h - y
        return (1/(2*n))*(np.dot(r, r))**2

def grad_desc(X,Y,beta,alpha,k,n,m):
    for i in range(k+1):
        temp = X  
        temp = np.transpose(temp)
        h = np.dot(X, beta)
        #print("hey this is h-y",h-Y)
        #beta = beta - (alpha/n)*np.sum(np.dot((h - Y), temp))
        beta = beta - (alpha/n)*((np.sum(np.transpose((h-Y)*np.transpose(X)),axis=0)))
        #print(costf(X,Y,beta,n,m))
    print("this is after descending the slope")
    print(beta)
        
    return beta

def lin_reg(X,Y,k,alpha):
    #add x_0 = 1
    #X[:,0] = 1
    m = len(X[0,:])
    n = len(X[:,0])
    X = np.append(np.ones(len(X[:,0]))[...,None],X,1)
    beta = np.zeros(m+1)
    beta = grad_desc(X,Y,beta,alpha,k,n,m)
    return beta #, cost(X,Y,beta,m,n)
    
beta,X,Y = gen(0.5,6,5)

k = 100000
alpha = 0.1
lin_reg(X,Y,k,alpha)
