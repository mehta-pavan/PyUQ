#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 10:53:58 2018

@author: pmehta
"""
import numpy as np
from matplotlib import pyplot as plt

class sin_test:
    
    """
    
    This scipt demonstartes how data is to be genereated and fed into cAPK algorithm for a 4 dimensional case
    
    
    """
    
    
    
    
    def test(T0 = 300, A1 = 10, A2 = 10, A3 = 10, A4 = 10, net_quad_point = 10, A1_range = 10,  A2_range = 10, A3_range = 10 ,  A4_range = 10, grid_poins = 100):
       
        """
        
        This method genereates the neceasry data for cAPK
        
        """
        
        
        x = np.linspace(0, np.pi, num = grid_poins, endpoint = True )
        
        A1 = np.linspace(A1 - A1_range/2, A1 + A1_range/2, num = net_quad_point, endpoint = True)
        A2 = np.linspace(A2 - A2_range/2, A2 + A2_range/2, num = net_quad_point, endpoint = True)
        A3 = np.linspace(A3 - A3_range/2, A3 + A3_range/2, num = net_quad_point, endpoint = True)
        A4 = np.linspace(A4 - A4_range/2, A4 + A4_range/2, num = net_quad_point, endpoint = True)
        
        f0 = sin_test.func(x, T0, ((A1[0] + A1[net_quad_point-1])/2), ((A2[0] + A2[net_quad_point-1])/2), ((A3[0] + A3[net_quad_point-1])/2), ((A4[0] + A4[net_quad_point-1])/2) )
        
        
    
    
        Quad_points = np.stack((A1, A2, A3, A4), axis = 1)
        
        f1, order1_index = sin_test.order1_terms(x, T0, Quad_points)
        
        f2, Quad_points_order2, order2_index = sin_test.order2_terms(x, T0, Quad_points)
        
        Quad_points_validation = np.stack((np.array((A1[0], A1[net_quad_point -1])), np.array((A2[0], A2[net_quad_point -1])), np.array((A3[0], A3[net_quad_point -1])), np.array((A4[0], A4[net_quad_point -1]))), axis = 1)
        
        data_validation = sin_test.data_valida(x, T0, Quad_points_validation, dim_max = 4)
        
        return f0, Quad_points, f1, order1_index, f2, Quad_points_order2, order2_index, data_validation, Quad_points_validation
    
    
    
    def func(x, T0, A1, A2, A3, A4):
        """
        Test function
        """
        
        
        y = T0 + A1*np.sin(x) + A2*np.sin(2*x) + A3*np.sin(3*x) + A4*np.sin(4*x)
       
        return y
   
    
    def data_valida(x, T0, Quad_points, dim_max = 4):
        
        """
        Generating validation data set
        
        """
        
        
        shape = x.shape
        grid_points = shape[0]
        
        data_validation = np.zeros((grid_points, 16))
        
        counter = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
    
                        data_validation[:,counter] = sin_test.func(x, T0, Quad_points[i,0], Quad_points[j,1], Quad_points[k,2], Quad_points[l,3])
                        counter += 1
                        
        return data_validation
        
   
    def order1_terms(x, T0, Quad_points, dim_max = 4):
        
        """
        
        Generating order 1 terms
        
        """
        
        
        
        shape = x.shape
        grid_points = shape[0]
        
        shape = Quad_points.shape
        net_quad_points, dim_max = shape[0], shape[1]
        
        f1 = np.zeros((grid_points, net_quad_points, dim_max))
        
        order1_index = np.zeros((net_quad_points, dim_max))
        
        for i in range(dim_max):
            
                       
            
            for j in range(net_quad_points):
                
                
                if i == 0:
                    A1 = Quad_points[j,i]
                    A2 = (Quad_points[0,1] + Quad_points[net_quad_points -1,1]) / 2
                    A3 = (Quad_points[0,2] + Quad_points[net_quad_points -1,2]) / 2
                    A4 = (Quad_points[0,3] + Quad_points[net_quad_points -1,3]) / 2
                
                elif i == 1:
                    A2 = Quad_points[j,i]
                    A1 = (Quad_points[0,0] + Quad_points[net_quad_points -1,0]) / 2
                    A3 = (Quad_points[0,2] + Quad_points[net_quad_points -1,2]) / 2
                    A4 = (Quad_points[0,3] + Quad_points[net_quad_points -1,3]) / 2
                    
                elif i == 2:
                    A3 = Quad_points[j,i]
                    A2 = (Quad_points[0,1] + Quad_points[net_quad_points -1,1]) / 2
                    A1 = (Quad_points[0,0] + Quad_points[net_quad_points -1,0]) / 2
                    A4 = (Quad_points[0,3] + Quad_points[net_quad_points -1,3]) / 2
                    
                elif i == 3:
                    A4 = Quad_points[j,i]
                    A2 = (Quad_points[0,1] + Quad_points[net_quad_points -1,1]) / 2
                    A3 = (Quad_points[0,2] + Quad_points[net_quad_points -1,2]) / 2
                    A1 = (Quad_points[0,0] + Quad_points[net_quad_points -1,0]) / 2
                                
            
                f1[:,j,i] = sin_test.func(x, T0, A1, A2, A3, A4)
                order1_index[j,i] = int(j)
                
                
                
        return f1, order1_index
    
    
    def order2_terms(x, T0, Quad_points, net_order2_terms = 6):
        
        
        """
        
        Generating order 2 data
        
        """
        
        
        
        
        shape = x.shape
        grid_points = shape[0]
        
        shape = Quad_points.shape
        net_quad_points_per_dim  = shape[0]
        
        f2 = np.zeros((grid_points, net_quad_points_per_dim**2, net_order2_terms))
        
        order2_index = np.zeros((net_quad_points_per_dim**2, 2, net_order2_terms))
        
        Quad_points_order2 = np.zeros((net_quad_points_per_dim, 2, net_order2_terms))
        
        for i in range(net_order2_terms):
            
            counter = 0
            for j in range(net_quad_points_per_dim):
                
                for k in range(net_quad_points_per_dim):
                    
                    
                      
                    if i == 0:
                        A1 = Quad_points[j,0]
                        A2 = Quad_points[k,1]
                        A3 = (Quad_points[0,2] + Quad_points[net_quad_points_per_dim - 1,2]) / 2
                        A4 = (Quad_points[0,3] + Quad_points[net_quad_points_per_dim - 1,3]) / 2
                       
                        if counter == 0:
                            Quad_points_order2[:,:, i] = np.stack((Quad_points[:,0], Quad_points[:,1]), axis = 1) 
                       
                    
                    elif i == 1:
                        A1 = Quad_points[j,0]
                        A2 = (Quad_points[0,1] + Quad_points[net_quad_points_per_dim - 1,1]) / 2
                        A3 = Quad_points[k,2]
                        A4 = (Quad_points[0,3] + Quad_points[net_quad_points_per_dim - 1,3]) / 2
                        
                       
                        if counter == 0:
                            Quad_points_order2[:,:, i] = np.stack((Quad_points[:,0], Quad_points[:,2]), axis = 1) 
                        
                    elif i == 2:
                       
                        A1 = Quad_points[j,0]
                        A2 = (Quad_points[0,1] + Quad_points[net_quad_points_per_dim - 1,1]) / 2
                        A3 = (Quad_points[0,2] + Quad_points[net_quad_points_per_dim - 1,2]) / 2
                        A4 = Quad_points[k,3]
                       
                        
                       
                        if counter == 0:
                            Quad_points_order2[:,:, i] = np.stack((Quad_points[:,0], Quad_points[:,3]), axis = 1) 
                        
                    elif i == 3:
                        A1 = (Quad_points[0,0] + Quad_points[net_quad_points_per_dim - 1,0]) / 2
                        A2 = Quad_points[j,1]
                        A3 = Quad_points[k,2]
                        A4 = (Quad_points[0,3] + Quad_points[net_quad_points_per_dim - 1,3]) / 2
                       
                        
                        if counter == 0:
                            Quad_points_order2[:,:, i] = np.stack((Quad_points[:,1], Quad_points[:,2]), axis = 1) 
                     
                        
                    elif i == 4:
                        A1 = (Quad_points[0,0] + Quad_points[net_quad_points_per_dim - 1,0]) / 2
                        A2 = Quad_points[j,1]
                        A3 = (Quad_points[0,2] + Quad_points[net_quad_points_per_dim - 1,2]) / 2
                        A4 = Quad_points[k,3]
                       
                       
                        if counter == 0:
                            Quad_points_order2[:,:, i] = np.stack((Quad_points[:,1], Quad_points[:,3]), axis = 1) 
                        
                    elif i == 5:
                        A1 = (Quad_points[0,0] + Quad_points[net_quad_points_per_dim - 1,0]) / 2
                        A2 = (Quad_points[0,1] + Quad_points[net_quad_points_per_dim - 1,1]) / 2
                        A3 = Quad_points[j,2]
                        A4 = Quad_points[k,3]
                        
                       
                        if counter == 0:
                            Quad_points_order2[:,:, i] = np.stack((Quad_points[:,2], Quad_points[:,3]), axis = 1) 
                        
                        
                    
                    f2[:,counter,i] = sin_test.func(x, T0, A1, A2, A3, A4)
                    order2_index[counter,:,i] = np.array((j,k), dtype = int)
                    
                    counter += 1
                    
        
        return f2, Quad_points_order2, order2_index
        
        
    