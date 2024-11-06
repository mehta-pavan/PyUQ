#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 17:21:45 2018

@author: pmehta
"""

import numpy as np


class kernel:
    
    """
    This class is reserved for definig kernels or co-variance fucntions for Krigging or otherwise"
    """
    
    def fit(X, discrestisation, Kernel = None, theta = 0.1):
        
        """
        This functions provides the Co-vaiance matrix
        
        Important Variable:
            Consider the function Y = F(X)
            X: Input, Type: array.
            discrestisation: Input, Type array
                discretitation of X, accoring to a rule. Refer 'sample.py'
            K: Kernel or Co-variance matrix of (X, X)
            K1: Kernel or Co-variance matrix of (X, discretitation)
            K2: Kernel or Co-variance matrix of (discretitation, discretitation)
        
        """
         
        # For Robustness 
        X = X.reshape(-1,1)
        
        discrestisation = discrestisation.reshape(-1,1)
         
         #Kernel Selection----------------------------------------------------------
        
        if ((Kernel == None) or (Kernel == "sq_exponential")): #use square kernel
            
                K = kernel.sq_exponential(X, X, theta)
                
                K1 = kernel.sq_exponential(X, discrestisation, theta)
                
                K2 = kernel.sq_exponential(discrestisation, discrestisation, theta)
        
        elif (Kernel == "poly_cubic_spline"):
            
                K = kernel.poly_cubic_spline(X,X, theta)
                                 
                K1 = kernel.poly_cubic_spline(X,discrestisation, theta)
                  
                K2 = kernel.poly_cubic_spline(discrestisation,discrestisation, theta)
                
        else:
            
            raise "Kernel selection Error: Permited value -> 'None', 'sq_exponential', 'poly_cubic_spline' "
        
        
                
        return K, K1, K2
  
    
    #----------------------------------------------------------------------------
    
    def sq_exponential(a, b, theta):
        
        """ GP squared exponential kernel """
        
        if theta is None:
            theta = 0.1
        
        kernelParameter = theta
        
        sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
        
        return np.exp(-.5 * (1/kernelParameter) * sqdist)


    #----------------------------------------------------------------------------

    def poly_cubic_spline(a,b, theta):
        
        """Polynomial Cubic spline kernel: Luca and Sagaut (2016)"""
        
        if theta is None:
            theta = 0.1
                
        def covariance(x1,y1, theta):
            
            """ 
            Covaraince function as per Luca and Sagaut (2016)
                
            The default value of theta = 0.1 as per article 
                
            Not suitable for N-dimensional problem.
            
            """
        
            dist = np.absolute(x1-y1)
        
            if dist < (0.5/theta):
                covar = 1 - 6*((dist*theta)**2) + 6*((dist*theta)**3)
            elif (0.5/theta) <= dist < (1/theta):
                covar = 2*((1 - (dist*theta))**3)
            elif dist >= (1/theta):
                covar = 0
            else:
                raise "Value Error: Check Cubic Spline Kernel"
                
            return covar
        
        
        #Computing Ranges 
        
        S1, S2 = a.shape, b.shape
        
        R_1, R_2 = S1[0], S2[0]
        
        
        # Buliding Kernel matix
        
        kernel = np.zeros((R_1,R_2))
                       
        for i in range(R_1):
            for j in range(R_2):
                kernel[i,j] = covariance(a[i],b[j], theta)
     
        
        return kernel



class test:
    
    """ This class is reserved for performing test of this file only"""
    
    def kernel_matrix(X, discrestisation, Kernel = None, theta = 0.1):
        
        return kernel.fit(X, discrestisation, Kernel, theta)
