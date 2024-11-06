#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 11:16:41 2018

@author: pmehta
"""
import cv as cv
import numpy as np
import krigging as krig
    
    
class leave_one_out:
    
    """
    
    This class perform Cross validation for 1 and 2 dimensional functiona.
    
    It is to be noted the response surface is generated using Krigging
    
    """
    
    def fit_1D(data_matrix, data_order1, Quad_points, discrestisation = None, Kernel = "poly_cubic_spline", theta = 0.1, auto_uniform_sampling = True, L1_sense = True, L2_sense = False):
        
        """
        This function perform Cross validation for 1 dimeniosnal functions
        
        """
        
        shape = data_matrix.shape
        
        grid_points, net_computational_points, dim_max = shape[0], shape[1], shape[2]
        
        shape = Quad_points.shape
        
        net_quad_points = shape[0]

        
        
        train_data = np.zeros((grid_points, net_computational_points))
        
        idx = np.arange(0, net_quad_points, 1)
        
        test_data = np.zeros((grid_points, net_computational_points, net_quad_points-2, dim_max))
        
        CV_score = np.zeros((grid_points, net_quad_points - 2, dim_max))
        
        
        for i in range(dim_max):
            
            counter = 0
            
            for j in range(net_quad_points):
         
                train_data[:,:] = data_matrix[:,:,i]
                
                if ((j == 0) or (j == (net_quad_points -1))):
                    pass
                else:
            
                    idxx = list(filter(lambda x : x != j, idx))
                    
                    for z in range(grid_points):
                
                        gen_test_data = data_order1[z, idxx, i]
                
                        index1 = Quad_points[idxx, i]
                
                        Quad_discard, test_data[z,:,counter,i], index_discard = krig.gaussion_process_regression.fit_1D(index1, gen_test_data, discrestisation, Kernel, theta, disceretistation_samples = net_computational_points)
                    
                    
                    if L1_sense == True:
                    
                        CV_score[:,counter,i] = cv.CV.global_l1(train_data, test_data[:,:,counter,i])
                        counter += 1
                    elif L2_sense == True:
                        CV_score[:,counter,i] = cv.CV.global_l2(train_data, test_data[:,:,counter,i])
                        counter += 1
                    else:
                        raise "Cross validation criteria: L1_sense or L2_sense must be True"
                    
                
        return CV_score, test_data
        
        


    def fit_2D(data_matrix, data_order2, Quad_points, index, discrestisation = None, Kernel = "poly_cubic_spline", theta = 0.1, auto_uniform_sampling = True, L1_sense = True, L2_sense = False):
        
        """
        This function perform Cross validation for 2 dimeniosnal functions
        
        """
        
        
        shape = Quad_points.shape
        
        net_Q_points_per_dim = shape[0]
               
        shape1 = data_matrix.shape
        
        grid_points, computational_points, net_order2_terms = shape1[0], shape1[1], shape1[2]
        
        
        if (auto_uniform_sampling == True):
            
            disceretistation_samples = int(np.sqrt(computational_points))
                        
            print("Automtatic Unifrom Sampling with samples per dim =", disceretistation_samples)
            
            disceretistation_2 = None
            disceretistation_1 = None                              
                                   
        else:
                        
            disceretistation_2 = discrestisation[:, 1]
            disceretistation_1 = discrestisation[:, 0]
        
        
        index1 = np.zeros((net_Q_points_per_dim*disceretistation_samples, 2), dtype = int)
        
        mu_all = np.zeros((grid_points, disceretistation_samples))
        
        CV_score = np.zeros((grid_points, net_Q_points_per_dim**2 - 4, net_order2_terms))
        
        test_data = np.zeros((grid_points, computational_points, net_Q_points_per_dim**2 - 4, net_order2_terms))
        
        for n in range(net_order2_terms):
            
            train_data = data_matrix[:,:,n]
            
            counter1 = 0
            
            for m in range(net_Q_points_per_dim):
                
                if ((m == 0) or (m == (net_Q_points_per_dim-1))):
                    pass
                
                else:
                
                    counter = 0
            
                    for i in range(net_Q_points_per_dim):
                        
                                
                                X = np.zeros((net_Q_points_per_dim))    
                    
                                #idx = np.zeros((net_Q_points_per_dim), dtype = int)
                    
                                Y = np.zeros((net_Q_points_per_dim))
                                
                                mu_1 = np.zeros((grid_points, disceretistation_samples))
                                
                                                     
                                idx = np.array((np.where((index[:,0, n] >= 0) & (index[:,1, n] == i))), dtype = int).reshape(-1)
                                
                                idx1 = np.arange(0, net_Q_points_per_dim, 1)
                                
                                
                                idxx1 = np.array((list(filter(lambda x : x != m, idx1))), dtype = int).reshape(-1)
                                
                    
                                
                                idxx = idx[idxx1]
                                 
                                for l in range(grid_points): 
                                     
                                    Y = data_order2[l,idxx, n]
                                     
                                    X = Quad_points[idxx1,0, n]
                                    
                                                             
                                    discrestisation1, mu, variance = krig.gaussion_process_regression.fit_1D(X,Y, disceretistation_1, Kernel, theta, disceretistation_samples)
                                     
                                    mu_1[l, :] = mu.reshape(-1)
                                
                                mu_all = np.hstack((mu_all, mu_1))
                                
                                for l in range(disceretistation_samples):
                            
                                   
                                    
                                    index1[counter, 0:] = np.array((l,i))
                                                       
                                    counter += 1
                                
                                
                                    
                    mu_all = mu_all[:,disceretistation_samples:]
                    
                    
                    
                    #Reating in 2rd dimension 
                    
                    
                    mu_all1 = np.zeros((grid_points, disceretistation_samples))
                    
                    
                    for j in  range(disceretistation_samples):
                           
                                
                                X = np.zeros((net_Q_points_per_dim))    
                    
                                #idx = np.zeros((net_Q_points_per_dim), dtype = int)
                    
                                Y = np.zeros((net_Q_points_per_dim))
                                
                                mu_1 = np.zeros((grid_points, disceretistation_samples))
                                
                                                     
                                idx = np.array((np.where((index1[:,0] == j) & (index1[:,1] >= 0) )), dtype = int).reshape(-1)
                                 
                                for l in range(grid_points): 
                                     
                                    Y = mu_all[l,idx]
                                     
                                    X = Quad_points[:,1,n]
                                     
                                                            
                                    discrestisation2, mu, variance = krig.gaussion_process_regression.fit_1D(X,Y, disceretistation_2, Kernel, theta, disceretistation_samples)
                                     
                                    mu_1[l, :] = mu.reshape(-1)
                                
                                mu_all1 = np.hstack((mu_all1, mu_1))
                                
                              
                                    
                    mu_all1 = mu_all1[:,disceretistation_samples:]
                    
                    test_data[:,:, counter1, n] = mu_all1
                    
                    if L1_sense == True:
                    
                        CV_score[:,counter1,n] = cv.CV.global_l1(train_data, mu_all1)
                        counter += 1
                    elif L2_sense == True:
                        CV_score[:,counter1,n] = cv.CV.global_l2(train_data, mu_all1)
                        counter += 1
                    else:
                        raise "Cross validation criteria: L1_sense or L2_sense must be True"
                    
                  
                    
                    
                    
                    
                    #removing a point in dim2------------------------------------------------------------------------------------------------------
                    
                    
                    counter = 0
            
                    for i in range(net_Q_points_per_dim):
                        
                                
                                X = np.zeros((net_Q_points_per_dim))    
                    
                                #idx = np.zeros((net_Q_points_per_dim), dtype = int)
                    
                                Y = np.zeros((net_Q_points_per_dim))
                                
                                mu_1 = np.zeros((grid_points, disceretistation_samples))
                                
                                                     
                                idx = np.array((np.where((index[:,0, n] >= 0) & (index[:,1, n] == i))), dtype = int).reshape(-1)
                                
                                                              
                    
                                
                                idxx = idx[idxx1]
                                 
                                for l in range(grid_points): 
                                     
                                    Y = data_order2[l,idx, n]
                                     
                                    X = Quad_points[:,0, n]
                                    
                                                             
                                    discrestisation1, mu, variance = krig.gaussion_process_regression.fit_1D(X,Y, disceretistation_1, Kernel, theta, disceretistation_samples)
                                     
                                    mu_1[l, :] = mu.reshape(-1)
                                
                                mu_all = np.hstack((mu_all, mu_1))
                                
                                for l in range(disceretistation_samples):
                            
                                   
                                    
                                    index1[counter, 0:] = np.array((l,i))
                                                       
                                    counter += 1
                                
                                
                                    
                    mu_all = mu_all[:,disceretistation_samples:]
                    
                    
                    
                    #Reating in 2rd dimension 
                    
                    
                    mu_all1 = np.zeros((grid_points, disceretistation_samples))
                    
                    
                    for j in  range(disceretistation_samples):
                           
                                
                                X = np.zeros((net_Q_points_per_dim))    
                    
                                #idx = np.zeros((net_Q_points_per_dim), dtype = int)
                    
                                Y = np.zeros((net_Q_points_per_dim))
                                
                                mu_1 = np.zeros((grid_points, disceretistation_samples))
                                
                                
                                                     
                                idx = np.array((np.where((index1[:,0] == j) & (index1[:,1] >= 0) )), dtype = int).reshape(-1)
                                
                                idx1 = np.arange(0, net_Q_points_per_dim, 1)
                                
                                
                                idxx1 = np.array((list(filter(lambda x : x != m, idx1))), dtype = int).reshape(-1)
                                
                                idxx = idx[idxx1]
                                
                                                     
                               
                                 
                                for l in range(grid_points): 
                                     
                                    Y = mu_all[l,idxx]
                                     
                                    X = Quad_points[idxx1,1,n]
                                     
                                                            
                                    discrestisation2, mu, variance = krig.gaussion_process_regression.fit_1D(X,Y, disceretistation_2, Kernel, theta, disceretistation_samples)
                                     
                                    mu_1[l, :] = mu.reshape(-1)
                                
                                mu_all1 = np.hstack((mu_all1, mu_1))
                                
                              
                                    
                    mu_all1 = mu_all1[:,disceretistation_samples:]
                    
                    test_data[:,:, counter1, n] = mu_all1
                    
                    if L1_sense == True:
                    
                        CV_score[:,counter1,n] = cv.CV.global_l1(train_data, mu_all1)
                        counter += 1
                    elif L2_sense == True:
                        CV_score[:,counter1,n] = cv.CV.global_l2(train_data, mu_all1)
                        counter += 1
                    else:
                        raise "Cross validation criteria: L1_sense or L2_sense must be True"
                    
    
    
    
    
        return CV_score, test_data

