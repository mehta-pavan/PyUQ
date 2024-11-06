#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 14:09:43 2018

@author: pmehta
"""

import anova_decomposition as an
import anova_terms as ant

import stat_var_expect as stat
import numpy as np


class adpative_anova:
    
    
     #----------------------------------------------------------------
        
        #seleetion creteria
        
    #---------------------------------------------------------------
                        
    
    def var_creteria_order1_terms(f0_data, data_order1, data_order2, order1_treshold = 0.9, order2_thershold = 0.9, adpative_in_order1 = False, trunc_order = 2):
        
        
        #parameter intiitisation
        shape = data_order1.shape
        
        grid_points, computational_points_order1, dim_max = shape[0], shape[1], shape[2]
        
        
        #Get data of varinace of order 1 termms and their sum 
        var_i, net_var_order1 = stat.variance.order1_terms(f0_data, data_order1)
        
        #Get the total varinace
        var = stat.variance.total(f0_data, data_order1, data_order2, trunc_order)
            
            
        #order 1 terms selection
        
        if (adpative_in_order1 == True):
            
            sumer = np.zeros((grid_points))
            
            select_order1 = np.zeros((grid_points))
            
            for i in range(dim_max):
                
                
                for z in range(grid_points):
                                     
                                       
                    sumer[z] = sumer[z] + (var_i[z,i])
                                      
                    
                select_order1 = sumer / net_var_order1
                
                
                
                
                #print(select_order1)
                
                if (np.all(select_order1 >= order1_treshold)):
                    
                    
                    print("the max order 1 terms needed = ", i, "thershold value =", order1_treshold)
                   
                    #Update net order 1 terms variance
                    #Use this variance if D1 is not N
                    #net_var_order1_updated = sumer
                    
                    active_dim = i
                    
                    break
                
                else:
                    
                    print("order 1 terms needed of dimeniosn = ", i, "for thershold value =", order1_treshold)
                    
                    #Use this variance if D1 is not N
                    #net_var_order1_updated = sumer
                    
                    active_dim = i
                   
                    pass
                
        else:
            
            #Use this variance if D1 is not N
            #net_var_order1_updated = net_var_order1
            
            active_dim = dim_max
            
        
        
        #Display Active dimension for order 2 terms
        print("")
        print("Active Dimension for order 2 terms =", active_dim, "Index count form 0")
        print("")
        
        
        
        return active_dim, var, var_i, net_var_order1
        
        
        
        
    def var_creteria_order2_terms(f0_data, data_order1, data_order2, order1_treshold = 0.9, order2_thershold = 0.9, adpative_in_order1 = False, trunc_order = 2):
        
        
        #parameter intiitisation                  
        shape = data_order1.shape
        
        grid_points, computational_points_order1, dim_max = shape[0], shape[1], shape[2]
            
        #Total number of order 2 terms
        net_order2_terms = int((dim_max*(dim_max -1)) - ((dim_max)*(dim_max-1)*0.5))
        
        
        #Get variance data
        
        var_ij = stat.variance.order2_terms(f0_data, data_order1, data_order2)
        
        active_dim, var, var_i, net_var_order1 = adpative_anova.var_creteria_order1_terms(f0_data, data_order1, data_order2, order1_treshold, order2_thershold, adpative_in_order1, trunc_order)
        
       
        
        
        
        
        #Order 2 terms selection --> of X_ij term   
        
        ignore_order2_index_var = []
       
        
        for i in range(net_order2_terms):
            
            sumer = np.zeros((grid_points))
            
            
            for z in range(grid_points):               
                
                #zero handler
                if net_var_order1[z] == 0:
                    net_var_order1[z] = 1e-10
                
                               
               
                
                sumer[z] = (var_ij[z,i] / net_var_order1[z])
                
                
               
                
                
              
                          
                  
            #secltion certeria  
                  
            if (np.all(sumer <= order2_thershold)):
                
                print(" The second order term to be ignored", i, "for thershold =", order2_thershold, "max value =", max(sumer))
                
                #f2_updated[:, :, :, i] = np.zeros((grid_points, computational_points_order1, computational_points_order1))
                
                ignore_order2_index_var.append(i)
                
                pass
            
            else:
                
                print(" The second order term needed", i, "for thershold =", order2_thershold, "max value =", max(sumer))
                
                               
                pass
                
                
        ignore_order2_index_var = np.array(ignore_order2_index_var)
        
        
        return active_dim, var_i, var_ij, net_var_order1, var, ignore_order2_index_var
    
  

class higher_order_selection:
    
    def order_2_selection(f0_data, data_order1, data_order2, data_validation, selection_thershold = 0.05):
        
        
        
               
        #get order 1 decoposition
        f_anova_v1, index1 = an.cANOVA_decomposition.select(f0_data, data_order1, None, trunc_order= 1)
        
        #get order 2 decoposition
        f_anova_v2, index = an.cANOVA_decomposition.select(f0_data, data_order1, data_order2, trunc_order= 2)
        
        #error coputation
        
        error_v2 = f_anova_v2 - data_validation
        
        error_v1 = f_anova_v1 - data_validation
               
              
        shape = f_anova_v1.shape
        
        grid_points, net_anova_terms = shape[0], shape[1]
        
        mean_error_v1, mean_error_v2, mean_an_v1, mean_an_v2 = np.zeros((grid_points)), np.zeros((grid_points)), np.zeros((grid_points)), np.zeros((grid_points))
        
        #mean error computation
        
        for z in range(grid_points):
            
            sumer1, sumer2, sumer3, sumer4 = 0, 0, 0, 0
        
            for i in range(net_anova_terms):
                            
               sumer1 = sumer1 + (error_v1[z,i])
               
               sumer2 = sumer2 + (error_v2[z,i])
               
               #computing the mean of ANOVA decompostion
               
               sumer3 = sumer3 + (f_anova_v1[z,i])
               
               sumer4 = sumer4 + (f_anova_v2[z,i])
               
            
            mean_error_v1[z] = sumer1 / net_anova_terms
            
            mean_error_v2[z] = sumer2 / net_anova_terms
            
            mean_an_v1[z] = sumer3 / net_anova_terms
            
            mean_an_v2[z] = sumer4 / net_anova_terms
    
        
        
        #standard deviation of the ANOVA and its error
         
        std_an_v1, std_an_v2, std_error_v1, std_error_v2 = np.zeros((grid_points)), np.zeros((grid_points)), np.zeros((grid_points)), np.zeros((grid_points))
         
        for z in range(grid_points):
            
            sumer1, sumer2, sumer3, sumer4 = 0, 0, 0, 0
        
            for i in range(net_anova_terms):
                            
               sumer1 = sumer1 + (f_anova_v1[z,i] - mean_an_v1[z])**2
               
               sumer2 = sumer2 + (f_anova_v2[z,i] - mean_an_v2[z])**2
               
                         
               sumer3 = sumer3 + (error_v1[z,i] - mean_error_v1[z])**2
               
               sumer4 = sumer4 + (error_v2[z,i] - mean_error_v2[z])**2
               
            #zero handler
            if sumer1 == 0:
                sumer1 = 1e-10
            if sumer2 == 0:
                sumer2 = 1e-10
             
               
            std_an_v1[z] = np.sqrt(sumer1 / net_anova_terms)
            std_an_v2[z] = np.sqrt(sumer2 / net_anova_terms)
            std_error_v1[z] = np.sqrt(sumer3 / net_anova_terms)
            std_error_v2[z] = np.sqrt(sumer4 / net_anova_terms)
        
        #computing the diffference, Refer Maghri and Saguat(2016) 
        relative_diff = (std_error_v2 / std_an_v2) - (std_error_v1 / std_an_v1)
        
        
        #Selection ceiteria
        
        if (np.all(relative_diff <= selection_thershold)):
            
            print("Order 2 decomposition NOT required")
            
            trunc_order = 1
            
            mean_ev = mean_error_v1
            
        else:
            
            print("Order 2 decomposition required")
            
            trunc_order = 2
            
            mean_ev = mean_error_v2
         
            
        return relative_diff, mean_ev, trunc_order
        
        
