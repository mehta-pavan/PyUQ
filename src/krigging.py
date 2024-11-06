#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 15:42:56 2018

@author: pmehta
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 17:21:18 2018

@author: pmehta
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 13:41:42 2018

@author: pmehta
"""
#from __future__ import division

import numpy as np
import matplotlib.pyplot as pl
import kernel
import sample as samp

"""
This file performs Gaussion Process Regression or Kriging in One Dimension only

TODO: N - Dimensional Krigging

"""

#---------------------------------------------------------------------------------

class gaussion_process_regression:
    
    """ This class is reserved for performing krigging """
  
      
    def fit_1D(X,Y, discrestisation = None, Kernel = None, theta = 0.1, disceretistation_samples = 20):
        
        """ 
        Krigging in 1D: X and Y = F(X) 
        
        Important variables:
            X: For x values, Type: 1D array
            Y: For F(x) values, Type: 1D array
            n_points: Specifiying the number of discretisataion points: Type: int
            discrestisation: array storing the discretised points
            variance: array storing varainaces at discretised points
            mu: array storing mean value at discretised points
        
        """
        
        #For robustness
        X = X.reshape(-1,1)
        Y = Y.reshape(-1,1)
        
        
        #Range of X or dataset
        R_1 = min(X) 
        R_2 = max(X)
        
        if discrestisation is None:
            
            #Unifrom discretistion with default values
            #Refer "sample.py"
            discrestisation, X_discard = samp.sampling.samples(X, n_points = disceretistation_samples)
                    
        else:
            
            #Sorting the array in ascending order and reshaping it
            #discrestisation = np.sort(samples, axis = None)
            #discrestisation = discrestisation.reshape(-1,1)
            discrestisation = discrestisation.reshape(-1,1)
       
        
        #Calling Kernel for krigging        
        K, K1 , K2 = kernel.kernel.fit(X, discrestisation, Kernel, theta)
        
        
        #Cholesky decompisition---------------------------------------------------
        L = np.linalg.cholesky(K)
        
                       
        # compute the mean at our test points -----------------------------------        
        Lk = np.linalg.solve(L, K1)
       
        mu = np.dot(Lk.T, np.linalg.solve(L, Y))
        
        
        # compute the variance at our test points---------------------------------        
        variance = np.diag(K2) - np.sum(Lk**2, axis=0)
        
                        
        return discrestisation, np.array(mu).reshape(-1), variance


    
    
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



    def decomposition_2D(f_data, Quad_points, index, discrestisation = None, Kernel = "poly_cubic_spline", theta = 0.1, disceretistation_samples = 20, auto_uniform_sampling = True):
        
        """
        This function perform Krigging in 2 dimensions ort
        
        """
        
        
        shape = Quad_points.shape
        
        net_Q_points_per_dim, dim_max = shape[0], shape[1]       
               
        shape1 = f_data.shape
        
        grid_points = shape1[0]
        
        
        if (auto_uniform_sampling == True):
                        
            print("Automtatic Unifrom Sampling with samples per dim =", disceretistation_samples)
            
            disceretistation_2 = None
            disceretistation_1 = None                              
                                   
        else:
                        
            disceretistation_2 = discrestisation[:, 1]
            disceretistation_1 = discrestisation[:, 0]
        
        
        
        counter = 0
        
        Quad_1 = np.zeros(((net_Q_points_per_dim*disceretistation_samples), dim_max))
        
        index1 = np.zeros((net_Q_points_per_dim*disceretistation_samples, dim_max), dtype = int)
        
        mu_all = np.zeros((grid_points, disceretistation_samples))
        
        for i in range(net_Q_points_per_dim):
            
                    
                    X = np.zeros((net_Q_points_per_dim))    
        
                    idx = np.zeros((net_Q_points_per_dim), dtype = int)
        
                    Y = np.zeros((net_Q_points_per_dim))
                    
                    mu_1 = np.zeros((grid_points, disceretistation_samples))
                    
                                         
                    idx[:] = np.array(np.where((index[:,0] >= 0) & (index[:,1] == i)))
                     
                    for l in range(grid_points): 
                         
                        Y = f_data[l,idx]
                         
                        X = Quad_points[:,0]
                         
                                                 
                        discrestisation1, mu, variance = gaussion_process_regression.fit_1D(X,Y, disceretistation_1, Kernel, theta, disceretistation_samples)
                         
                        mu_1[l, :] = mu.reshape(-1)
                    
                    mu_all = np.hstack((mu_all, mu_1))
                    
                    for l in range(disceretistation_samples):
                        
                        Quad_1[counter, 0:] = np.array((np.round((discrestisation1[l]), decimals = 3), (Quad_points[i,1])))
                        
                        index1[counter, 0:] = np.array((l,i))
                                           
                        counter += 1
                        
        mu_all = mu_all[:,disceretistation_samples:]
        
        
        
        #Reating in 2rd dimension 
        
        
        
        counter = 0
        
        Quad_2 = np.zeros(((disceretistation_samples**2), dim_max))
        
        index2 = np.zeros(((disceretistation_samples**2), dim_max))
        
        mu_all1 = np.zeros((grid_points, disceretistation_samples))
        
        
        for j in  range(disceretistation_samples):
               
                    
                    X = np.zeros((net_Q_points_per_dim))    
        
                    idx = np.zeros((net_Q_points_per_dim), dtype = int)
        
                    Y = np.zeros((net_Q_points_per_dim))
                    
                    mu_1 = np.zeros((grid_points, disceretistation_samples))
                    
                                         
                    idx[:] = np.array(np.where((index1[:,0] == j) & (index1[:,1] >= 0) ))
                     
                    for l in range(grid_points): 
                         
                        Y = mu_all[l,idx]
                         
                        X = Quad_points[:,1]
                         
                                                
                        discrestisation2, mu, variance = gaussion_process_regression.fit_1D(X,Y, disceretistation_2, Kernel, theta, disceretistation_samples)
                         
                        mu_1[l, :] = mu.reshape(-1)
                    
                    mu_all1 = np.hstack((mu_all1, mu_1))
                    
                    for l in range(disceretistation_samples):
                        
                        Quad_2[counter, 0:] = np.array(((Quad_1[j,0]), np.round((discrestisation2[l]), decimals = 3) ))
                        
                        index2[counter, 0:] = np.array((j,l))
                                           
                        counter += 1
                        
        mu_all1 = mu_all1[:,disceretistation_samples:]
        
        
        Quad = np.zeros((disceretistation_samples, 2))
        
        for i in range(disceretistation_samples):
            
            Quad[i,0:] = np.array(( np.round((discrestisation1[i]), decimals = 4), np.round((discrestisation2[i]), decimals = 4) )).reshape(-1)
    
    
        return mu_all1, Quad, index2




#---------------------------------------------------------------------------------------------------------------------------------------------------------------


    def decomposition_4D(f_data, Quad_points, index, discrestisation = None, Kernel = "poly_cubic_spline", theta = 0.1, disceretistation_samples = 20, auto_uniform_sampling = True):
        
        """
                       
        
        
        """
        
        disceretistation_samples = int(disceretistation_samples)
        
        shape = Quad_points.shape
        
        net_Q_points_per_dim, dim_max = int(shape[0]), int(shape[1])
        
        if (auto_uniform_sampling == True):
                        
            print("Automtatic Unifrom Sampling with samples per dim =", disceretistation_samples)
            
            disceretistation_4 = None
            disceretistation_3 = None
            disceretistation_2 = None
            disceretistation_1 = None                              
                                   
        else:
            
            disceretistation_4 = discrestisation[:, 3]
            disceretistation_3 = discrestisation[:, 2]
            disceretistation_2 = discrestisation[:, 1]
            disceretistation_1 = discrestisation[:, 0]
            
            
        
        shape1 = f_data.shape
        
        grid_points = shape1[0]
        
        counter = 0
        
        Quad_1 = np.zeros((int((net_Q_points_per_dim**3)*disceretistation_samples), dim_max))
        
        index1 = np.zeros((int((net_Q_points_per_dim**3)*disceretistation_samples), dim_max), dtype = int)
        
        mu_all = np.zeros((grid_points, disceretistation_samples))
        
        for i in range(net_Q_points_per_dim):
            for j in range(net_Q_points_per_dim):
                for k in range(net_Q_points_per_dim):
                    
                    X = np.zeros((net_Q_points_per_dim))    
        
                    idx = np.zeros((net_Q_points_per_dim), dtype = int)
        
                    Y = np.zeros((net_Q_points_per_dim))
                    
                    mu_1 = np.zeros((grid_points, disceretistation_samples))
                    
                                         
                    idx = np.array((np.where((index[:,0] >= 0) & (index[:,1] == i) & (index[:,2] == j) & (index[:,3] == k))), dtype = int)
                     
                    for l in range(grid_points): 
                         
                        Y = f_data[l,idx]
                         
                        X = Quad_points[:,0]
                         
                                                 
                        discrestisation1, mu, variance = gaussion_process_regression.fit_1D(X,Y, disceretistation_1, Kernel, theta, disceretistation_samples)
                         
                        mu_1[l, :] = mu.reshape(-1)
                    
                    mu_all = np.hstack((mu_all, mu_1))
                    
                    for l in range(disceretistation_samples):
                        
                        Quad_1[counter, 0:] = np.array((np.round((discrestisation1[l]), decimals = 3), (Quad_points[i,1]), (Quad_points[j,2]) , (Quad_points[k,3])))
                        
                        index1[counter, 0:] = np.array((l,i,j,k))
                                           
                        counter += 1
                        
        mu_all = mu_all[:,disceretistation_samples:]
        
        
        
        #Reating in 3rd dimension 
        
        
        
        counter = 0
        
        Quad_2 = np.zeros((((net_Q_points_per_dim**2)*(disceretistation_samples**2)), dim_max))
        
        index2 = np.zeros((((net_Q_points_per_dim**2)*(disceretistation_samples**2)), dim_max))
        
        mu_all1 = np.zeros((grid_points, disceretistation_samples))
        
        for i in range(net_Q_points_per_dim):
            for j in  range(net_Q_points_per_dim):
                for k in range(disceretistation_samples):
                    
                    X = np.zeros((net_Q_points_per_dim))    
        
                    #idx = np.zeros((net_Q_points_per_dim), dtype = int)
        
                    Y = np.zeros((net_Q_points_per_dim))
                    
                    mu_1 = np.zeros((grid_points, disceretistation_samples))
                    
                                         
                    idx = np.array(np.where((index1[:,0] == k) & (index1[:,1] >= 0) & (index1[:,2] == i) & (index1[:,3] == j)))
                     
                    for l in range(grid_points): 
                         
                        Y = mu_all[l,idx]
                         
                        X = Quad_points[:,1]
                         
                                                
                        discrestisation2, mu, variance = gaussion_process_regression.fit_1D(X,Y, disceretistation_2, Kernel, theta, disceretistation_samples)
                         
                        mu_1[l, :] = mu.reshape(-1)
                    
                    mu_all1 = np.hstack((mu_all1, mu_1))
                    
                    for l in range(disceretistation_samples):
                        
                        Quad_2[counter, 0:] = np.array((np.round((discrestisation1[k]), decimals = 3),  np.round((discrestisation2[l]), decimals = 3), (Quad_points[i,2]), (Quad_points[j,3])))
                        
                        index2[counter, 0:] = np.array((k, l, i, j))
                                           
                        counter += 1
                        
        mu_all1 = mu_all1[:,disceretistation_samples:]
        
        
            
              
        
        #Reating in 2rd dimension 
        
        
        
        counter = 0
        
        Quad_3 = np.zeros((int((net_Q_points_per_dim)*(disceretistation_samples**3)), dim_max))
        
        index3 = np.zeros((int((net_Q_points_per_dim)*(disceretistation_samples**3)), dim_max))
        
        mu_all2 = np.zeros((grid_points, disceretistation_samples))
        
        for i in range(net_Q_points_per_dim):
            for j in  range(disceretistation_samples):
                for k in range(disceretistation_samples):
                    
                    X = np.zeros((net_Q_points_per_dim))    
        
                    i#dx = np.zeros((net_Q_points_per_dim), dtype = int)
        
                    Y = np.zeros((net_Q_points_per_dim))
                    
                    mu_1 = np.zeros((grid_points, disceretistation_samples))
                    
                                         
                    idx = np.array(np.where((index2[:,0] == j) & (index2[:,1] == k) & (index2[:,2] >= 0) & (index2[:,3] == i)))
                     
                    for l in range(grid_points): 
                         
                        Y = mu_all1[l,idx]
                         
                        X = Quad_points[:,2]
                         
                                                 
                        discrestisation3, mu, variance = gaussion_process_regression.fit_1D(X,Y, disceretistation_3, Kernel, theta, disceretistation_samples)
                         
                        mu_1[l, :] = mu.reshape(-1)
                    
                    mu_all2 = np.hstack((mu_all2, mu_1))
                    
                    for l in range(disceretistation_samples):
                        
                        Quad_3[counter, 0:] = np.array((np.round((discrestisation1[j]), decimals = 3) , np.round((discrestisation2[k]), decimals = 3),  np.round((discrestisation3[l]), decimals = 3), (Quad_points[i,3])))
                        
                        index3[counter, 0:] = np.array((j,k,l,i))
                                           
                        counter += 1
                        
        mu_all2 = mu_all2[:,disceretistation_samples:]
            
        
                     
        
        #Reating in 2rd dimension 
        
        
        
        counter = 0
        
        Quad_4 = np.zeros((int(disceretistation_samples**4), dim_max))
        
        index4 = np.zeros((int(disceretistation_samples**4), dim_max))
        
        mu_all3 = np.zeros((grid_points, disceretistation_samples))
        
        for i in range(disceretistation_samples):
            for j in  range(disceretistation_samples):
                for k in range(disceretistation_samples):
                    
                    X = np.zeros((net_Q_points_per_dim))    
        
                    #idx = np.zeros((net_Q_points_per_dim), dtype = int)
        
                    Y = np.zeros((net_Q_points_per_dim))
                    
                    mu_1 = np.zeros((grid_points, disceretistation_samples))
                    
                                        
                    idx = np.array(np.where((index3[:,0] == i) & (index3[:,1] == j) & (index3[:,2] == k) & (index3[:,3] >= 0)))
                     
                    for l in range(grid_points): 
                         
                        Y = mu_all2[l,idx]
                         
                        X = Quad_points[:,3]
                         
                                                 
                        discrestisation4, mu, variance = gaussion_process_regression.fit_1D(X,Y, disceretistation_4, Kernel, theta, disceretistation_samples)
                         
                        mu_1[l, :] = mu.reshape(-1)
                    
                    mu_all3 = np.hstack((mu_all3, mu_1))
                    
                    for l in range(disceretistation_samples):
                        
                        Quad_4[counter, 0:] = np.array((np.round((discrestisation1[i]), decimals = 3), np.round((discrestisation2[j]), decimals = 3) , np.round((discrestisation3[k]), decimals = 3),  np.round((discrestisation4[l, 0]), decimals = 3)))
                        
                        index4[counter, 0:] = np.array((i,j,k,l))
                                           
                        counter += 1
                        
        mu_all3 = mu_all3[:,disceretistation_samples:]
            



        return mu_all3, index4, Quad_4
















































