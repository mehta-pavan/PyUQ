#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 18:40:31 2018

@author: pmehta
"""

import numpy as np
import anova_terms as ant



class cANOVA:
    
    
         
    def decomposition_4D_v1(f0_data, data_order1):
            
        """
        This function computes the anchored ANOVA decompisition in 4 dimension with Trncation dimension = 1
        
        
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
            f_anova: ANOVA decomosition functional value; Type - Array 2 dimension
            index: Tracks the data; Type - Array 2 dimension
                  
                
        """ 
        
        #Calling functianl value at the anchored pooint
        f0 = ant.cANOVA_terms.compute_f0(f0_data)
        
        #get order 1 terms and neccessary data
        
        f1, grid_points, computational_points_order1, dim_max = ant.cANOVA_terms.order_1_terms(f0_data, data_order1)
        
               
        #Ininitlisation of array for ANOVA decomposition results 
        net_anova_terms = computational_points_order1**dim_max
        
        f_anova = np.zeros((grid_points, net_anova_terms))
        
        
        #index for tracking data
        index = np.zeros((net_anova_terms, dim_max))
        
        counter = 0
        
       
        
        #Numer of loops is equal to dimension of the problem
        #ADD more numer of loops as dimension increases
        for i in range(computational_points_order1):
            for j in range(computational_points_order1):
                for k in range(computational_points_order1):
                    for l in range(computational_points_order1):
                        
                        for z in range(grid_points):
                        
                            #Anova Expansion
                            
                            f_anova[z, counter] = f0[z]+ f1[z,i,0] + f1[z,j,1] + f1[z,k,2] + f1[z,l,3]
                                                                                   
                        index[counter, 0:] = [i, j, k , l]
                        
                        counter += 1
        

        return f_anova, index
    
    
    def decomposition_4D_v2(f0_data, data_order1, data_order2, ignore_order2_index = None):
            
        """
        This function computes the anchored ANOVA decompisition in 4 dimension with Trncation dimension = 2
        
        
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
            f_anova: ANOVA decomosition functional value; Type - Array 2 dimension
            index: Tracks the data; Type - Array 2 dimension
                  
                
        """ 
        
        #Calling functianl value at the anchored pooint
        f0 = ant.cANOVA_terms.compute_f0(f0_data)
        
        #get order 1 terms and neccessary data
        
        f1, grid_points, computational_points_order1, dim_max = ant.cANOVA_terms.order_1_terms(f0_data, data_order1)
        
        #get order 2 terms and necessary data
        
        f2, computational_points_order2, net_order2_terms, order2_index = ant.cANOVA_terms.order_2_terms(f0_data, data_order1, data_order2)
        
        #Ininitlisation of array for ANOVA decomposition results 
        
        net_anova_terms = computational_points_order1**dim_max
        
        
        #get order 1 decoposition
        #f_anova_v1, index1 = decomposition_4D_v1(f0_data, data_order1)
        
            
            
        f_anova_v2 = np.zeros((grid_points, net_anova_terms))
        
        
        #index for tracking data
        index = np.zeros((net_anova_terms, dim_max))
        
        counter = 0
        
        
        #For adpative anova - fill the not needed terms with zeros
        if np.any(ignore_order2_index != None):
            
            for i in range(net_order2_terms):
                
                if np.any(ignore_order2_index == i):
                    
                    print("Igonring order 2 term in anoava decompposition =", i)
                    
                    f2[:,:,:,i] = np.zeros((grid_points, computational_points_order1, computational_points_order1))
                else:
                    pass
        else:
            pass
        
        
        
       
        
        #Numer of loops is equal to dimension of the problem
        #ADD more numer of loops as dimension increases
        for i in range(computational_points_order1):
            for j in range(computational_points_order1):
                for k in range(computational_points_order1):
                    for l in range(computational_points_order1):
                        
                        for z in range(grid_points):
                            
                        
                            #Anova Expansion
                            
                            f_anova_v2[z, counter] = f0[z]+ f1[z,i,0] + f1[z,j,1] + f1[z,k,2] + f1[z,l,3] + f2[z,i,j,0] +  f2[z,i,k,1] +  f2[z,i,l,2] +  f2[z,j,k,3] +  f2[z,j,l,4] +  f2[z,k,l,5]
                                                                                   
                        index[counter, 0:] = [i, j, k , l]
                        
                        counter += 1
        

        return f_anova_v2, index