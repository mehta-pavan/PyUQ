#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 15:16:17 2018

@author: pmehta
"""
import anova_decomposition as an
import anova_terms as ant
import numpy as np




class variance:
    
    
    
    
    def total(f0_data, data_order1, data_order2, trunc_order = 2):
        
        shape = data_order1.shape
        
        grid_points = shape[0]
          
        #Calling functianl value at the anchored pooint
        f0 = ant.cANOVA_terms.compute_f0(f0_data)
        
          
        #Call ANOVA decomposition
        f_anova, index = an.cANOVA_decomposition.select(f0_data, data_order1, data_order2, trunc_order)
        
        shape = f_anova.shape
        
        net_anova_terms = shape[1]
        
        #compute sum of total variance 
        
        var = np.zeros((grid_points))  
        
                    
        for z in range(grid_points):
            
            sumer = 0
            
            for i in range(net_anova_terms):
        
                sumer = sumer + (f_anova[z,i] - f0[z])**2
                
            var[z] = sumer / net_anova_terms
        
        return var
        
    
    
    
    
    
    def order1_terms(f0_data, data_order1):
        
        
        #get order 1 terms and neccessary data
        
        f1, grid_points, computational_points_order1, dim_max = ant.cANOVA_terms.order_1_terms(f0_data, data_order1)
         
        #Compute order 1 variance
        
        var_i = np.zeros((grid_points, dim_max))
        
        for i in range(dim_max):
                        
            for z in range(grid_points):
                
                sumer = 0
                
                for j in range(computational_points_order1):
                
                    sumer = sumer + (f1[z,j,i]**2)                    
                    
                var_i[z,i] = sumer / computational_points_order1
                
                            
        # computing total order 1 variance
        
        net_var_order1 = np.zeros((grid_points))
        
        for z in range(grid_points):
                
            sumer1 = 0
            
            for i in range(dim_max):
                
                sumer1 = sumer1 + var_i[z,i]
            
            net_var_order1[z] = sumer1 / dim_max
        
        return var_i, net_var_order1
    
    
    
    
  
    
    
    def order2_terms(f0_data, data_order1, data_order2):
        
        
        #get order 2 terms and necessary data
        
        f2, computational_points_order2, net_order2_terms, order2_index = ant.cANOVA_terms.order_2_terms(f0_data, data_order1, data_order2)
        
        shape = data_order1.shape
        
        
        grid_points, computational_points_order1 = shape[0], shape[1]
        
        #compute variance order 2 terms
        
        var_ij = np.zeros((grid_points, net_order2_terms))
        
        
        for i in range(net_order2_terms):
            
            for z in range(grid_points):
                
                sumer = 0
                
                for j in range(computational_points_order1):
                    
                    for k in range(computational_points_order1):
                        
                        sumer = sumer + (f2[z,j,k,i]**2)
                        
                        
                var_ij[z,i] = sumer / computational_points_order2
                
                
        return var_ij
    
    
    
    
    
  





