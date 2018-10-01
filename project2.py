# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 15:04:13 2018

@author: Andrew
"""
    import numpy as np
    import numdifftools as nd
    import copy

def rosen(x):
     """The Rosenbrock function"""
     return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)


class problem():
    
    def __init__(self, func, compute_gradient = False):
        
        self.func = func
        
        if compute_gradient == True:
            
            # Calculate derivative func
            self.gradient = derivative()
            
    

class optimisation():
    
    def __init__(self,):
        
        g = self.func
        epsilon = 0.001
        
        # Choose start value (initial guess)
        
        x_0 = # Need the line-search method for this
        
        # Loop over k until a termination criterion is fulfilled:
        k = 0
        while g(x[k]) > epsilon
            x[k+1] = x[k] - G_inv[x[k]] * g(x[k])
            k = k + 1
            
            

%%%%%%%%%%%%%%%%%

def sq(x):
    return(x**2)
    
def derivative(func):
    h = 0.00001
    def der(x):
        return (func(x + h) - func(x)) / h
        
    return der




%%%%%%%%%%%%%%%%%
# Attempt to give approximation of smallest h that can be detected by computer
h1 = 0.00001
while (2 + h1 - 2) > 0:
        h = h1
        h1 = 0.01*h
%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def prod(x,y):
    return(x*y)
    
def fun(x):
    return (x)**2

def roots(func, x_0):
    x = x_0
    x1 = 0
    epsilon = 0.0000000001
    func_prime = derivative(func)
    while func(x) > epsilon:
        x1 = x - (func(x) / func_prime(x))
        x = x1
        print(x)

"""
def newt(func, x_0):
    x = x_0
    x1 = 0
    epsilon = 0.0000000001
    func_prime = derivative(func)
    func_prime2 = derivative(func_prime)
    while func(x) > epsilon:
        x1 = x - (func_prime(x) / func_prime2(x))
        x = x1
        print(x)
"""

def grad(func, x):
    
    h = 0.00001
    gradient = [0]*len(x)
    
    for ii in range(len(x)):
        x_h = copy.deepcopy(x)
        x_h[ii] = x_h[ii] + h
        gradient[ii] =  (func(x_h[ii]) - func(x)) / h
        
    return gradient

#####

x = np.array([3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 10.0])

def hess(func, x):
    h = 0.00001
    gradient = np.zeros((len(x), len(x)))
    
    for ii in range(len(x)):
        for jj in range(len(x)):
            x_h = copy.deepcopy(x)
            x_h[ii] = x_h[ii] + h
            x_h[jj] = x_h[jj] + h
            
            x_h1 = copy.deepcopy(x)
            x_h1[ii] = x_h1[ii] + h
            
            x_h2 = copy.deepcopy(x)
            x_h2[jj] = x_h2[jj] + h
            
            gradient[ii,jj] =  (func(x_h) - func(x_h1) - func(x_h2) + func(x)) / h**2
    return gradient
#####

        
def fun(x):
    return (x[0]**2) * (x[1]**2)

x = np.array([3.0,4.0])

def derivative(func, x):
    
    h = 0.00001
    gradient = [0]*len(x)
    for ii in range(len(x)):
        x_h = copy.deepcopy(x)
        x_h[ii] = x_h[ii] + h
        gradient[ii] =  (func(x_h) - func(x)) / h
    return gradient

def derivative(func, x):
    # x = [x1, x2,..., xn]
    h = 0.00001
    return (func(x1 + h, x2) - func(x1, x2)) / h
    
<<<<<<< HEAD
" Hell yeah"
=======
" Hello Ville"
>>>>>>> c157e8f83d8b8e130ab4433354f3606fa6eeb83c

H = nd.Hessian(fun)
g = nd.Gradient(fun)
alpha = 0.1

x = 2
y = 2

for i in range(1000):
    x1 = x - alpha * np.dot(np.linalg.inv(H(x,y)), g(x,y))
    x = x1


"Hello"

" Notepad editor added?"

" hello101"