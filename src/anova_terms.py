#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 18:26:44 2018

@author: pmehta
"""

import numpy as np


class cANOVA_terms:
    
    """
    This class performs Anchored ANOVA decomposition in 4-dimesnions with trucation dimension = 2
           
    """
        
    def compute_f0(f0_data):
        
        """T
        This functions gets the value at anchor points
        
        Inupt Data:
            f0_data: CFD data achor point: Type - Array 1 or 2 dimension
            
        Output:
            Functional value at achor point
        
        """
                       
        return  (np.array((f0_data)).reshape(-1,1))
    
    
    #---------------------------------------------------------------------------------------------
    
    
    def order_1_terms(f0_data, data_order1):
        
        """
        This function computes the first order terms required for anchored ANOVA decompisition
        
        Inupt Data:
            f0_data: CFD data achor point: Type - Array 1 or 2 dimension
            data_order1: CFD data for First order terms: Type - Array 3 dimension   
        
        
        Variables and ouput:
            grid_points: Total number of grid pints in the CFD Mesh; Type - int
            computational_points: Total number of Quadtarture points aloang the Hyperline; Type - int
            dim_max: Maximum dimension of problem; Type - int
            f0: Functional value at achor point: Type - Array 1 or 2 dimension
            f1: First order terms: Type - Array 3 dimension        
        
        """
        
        #Variable intialisation
        shape = data_order1.shape
        
        grid_points, computational_points, dim_max = shape[0], shape[1], shape[2]
        
        #Calling functianl value at the anchored pooint
        f0 = cANOVA_terms.compute_f0(f0_data)
        
        
        #order 1 terms computation---------------------------------
        
        f1 = np.zeros((grid_points, computational_points, dim_max))
        
        for k in range(dim_max):
            for j in range(computational_points):
                for i in range(grid_points):
                    
                    f1[i,j,k] = data_order1[i,j,k] - f0[i]
                                             
        
        return f1, grid_points, computational_points, dim_max
    
    
    #---------------------------------------------------------------------------------------------
    
    
    def order_2_terms(f0_data, data_order1, data_order2):
        
        """
        This function computes the second order terms required for anchored ANOVA decompisition
        
        
        Inupt Data:
            f0_data: CFD data achor point: Type - Array 1 or 2 dimension
            data_order1: CFD data for First order terms: Type - Array 3 dimension
            data_order2: CFD data for Second order terms: Type - Array 3 dimension 
        
        
        Variables and ouput:
            grid_points: Total number of grid pints in the CFD Mesh; Type - int
            computational_points_order1: Total number of Quadtarture points aloang the Hyperline; Type - int
            computational_points_order2: Total number of Quadtarture points aloang the Hyperplane; Type - int
            net_order2_terms: Total number of order 2 terms; Type - int
            dim_max: Maximum dimension of problem; Type - int
            f0: Functional value at achor point: Type - Array 1 or 2 dimension
            f1: First order terms: Type - Array 3 dimension
            f2: Second order terms: Type - Array 4 dimension
                             
                
        """
        
        #get mean value
        
        f0 = cANOVA_terms.compute_f0(f0_data)
        
        #get order 1 terms and neccessary data
        
        f1, grid_points, computational_points_order1, dim_max = cANOVA_terms.order_1_terms(f0_data, data_order1)
        
                
        #precautinary step
        if dim_max < 2:
            
            #raising error
            raise "Cannot compute ANOVA terms for less than 2 dimensions, if the experiment is in 2 or higher dim. then check data_order1 and data_order2 arrays"
        
        else:
             
            #------------------------------------------------------------------
            #Compute order 2 terms
            #------------------------------------------------------------------
            
                #Total number of order 2 terms
                net_order2_terms = int((dim_max*(dim_max -1)) - ((dim_max)*(dim_max-1)*0.5))
                
                shape = data_order2.shape
                computational_points_order2 = shape[1]
                
                #Intitailsing array for storage of order 2 terms
                
                
                #f2 = np.zeros((grid_points, computational_points_order1, net_order2_terms*2))
                f2 = np.zeros((grid_points, computational_points_order1, computational_points_order1, net_order2_terms))
                
                #intialising values for the loop below
                
                order2, order2_1, counter, dim1, dim2, counter1 = -1, -1, 0, 1, 2, 0
                
                order2_index = np.zeros((net_order2_terms, 2))
                
                #Computing order 2 terms
                
                while dim1 < dim_max:
                    dim2  = 2 
                    while dim2 <= dim_max:
                        if dim2 <= dim1:
                            pass
                
                        else:
                                    order2_index[counter1, 0:] = np.array((dim1-1, dim2-1), dtype = int) 
                                    order2 += 1
                                    counter = 0
                                    for i in range(computational_points_order1):
                                        #counter1 = 0
                                        order2_1 += 1
                                        for j in range(computational_points_order1):
                                            for m in range(grid_points):
                                                                               
                                                
                                                f2[m, i, j, order2] = data_order2[m, counter, order2] - f1[m,i,dim1-1] - f1[m,j,dim2-1] - f0[m]
                                                
                                                
                                                                
                                            counter += 1
                                    counter1 += 1
                        dim2 += 1
                    dim1 += 1
            
                 
           
        return f2, computational_points_order2, net_order2_terms, order2_index
    
    
    
  
    
