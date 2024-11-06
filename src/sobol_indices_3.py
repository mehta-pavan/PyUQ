#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 17:40:46 2018

@author: Pavan Mehta

Email: mehtapavanp@gmail.com
"""


import numpy as np
import krigging as krig
import sample
import krig_for_anova as generate_anova_terms
import anova_decomposition as an

class sobol_indices:
    
    """
    This class computes Sobol indices and statistics of the solution in N-dimesnion
    
    Note: The values at each sample point is obainted using Krigging using Polynomial cubic spline Kernel (Refer: Margheri and Saguat (2016) for kernel function)
    
    Quasi Monte-Carlo is achichved by generating samples from Sobol Sequence 
    
    """
    
    def indices(f0_data, data_order1, data_order2, Quad_data_order1, Quad_points_order2, order_2_index, trunc_order = 2, ignore_order2_index = None, Kernel = "poly_cubic_spline", theta = 0.1, discretisation_samples = 20):
        
        shape = data_order1.shape
        
        grid_points, compututational_points_order1, dim_max = shape[0], shape[1], shape[2]
         
        #Total number of order 2 terms
        net_order2_terms = int((dim_max*(dim_max -1)) - ((dim_max)*(dim_max-1)*0.5))
            
        
        discretisation = np.zeros((discretisation_samples, 2, dim_max))

        for i in range(dim_max):
            
            
                
                X = Quad_data_order1[:,i]
                                
                #get samples from sobol sequence
                discretisation[:,:,i], x_discard = sample.sampling.samples(X, sampling = "sobol", dim_max = 2, n_points = discretisation_samples)
                
       
        
        #computing order 1 and order 2 terms for anova on sobol sampling (omega1 and omega2)
        
        
        podk_data_order1_omega1, Quad_order1_omega1, index_order1_omega1 = generate_anova_terms.built_data.generating_order1_terms(data_order1, Quad_data_order1, discretisation[:,0, :], Kernel, theta, discretisation_samples)
        
        podk_data_order1_omega2, Quad_order1_omega2, index_order1_omega2 = generate_anova_terms.built_data.generating_order1_terms(data_order1, Quad_data_order1, discretisation[:,1, :], Kernel, theta, discretisation_samples)
        
        
        discretisation_order2_omega1 = np.zeros((discretisation_samples, 2, net_order2_terms))
        
        discretisation_order2_omega2 = np.zeros((discretisation_samples, 2, net_order2_terms))       
             
        
        
      
            
                          
        order2, dim1, dim2 = -1, 1, 2
            
        while dim1 < dim_max:
            dim2  = 2 
            while dim2 <= dim_max:
                if dim2 <= dim1:
                    pass
                
                else:
                            
                    order2 += 1
                            
                    discretisation_order2_omega1[:,:,order2] = np.stack((discretisation[:,0, dim1-1], discretisation[:,0, dim2-1]), axis = 1)
            
                    discretisation_order2_omega2[:,:,order2] = np.stack((discretisation[:,1, dim1-1], discretisation[:,1, dim2-1]), axis = 1)
                        
                            
                                            
                dim2 += 1
            dim1 += 1
                   
        
        
         
        podk_data_order2_omega1, Quad_order2_omega1, index_order2_omega1 = generate_anova_terms.built_data.generating_order2_terms(data_order2, Quad_points_order2, order_2_index, discretisation_order2_omega1, Kernel, theta, discretisation_samples, False)
                
        
        podk_data_order2_omega2, Quad_order2_omega2, index_order2_omega2 = generate_anova_terms.built_data.generating_order2_terms(data_order2, Quad_points_order2, order_2_index, discretisation_order2_omega2, Kernel, theta, discretisation_samples, False)
        
        
        
        
        
        
        
        
        
        
        
       
        
        podk_data_order2_omega12 = np.zeros((grid_points, discretisation_samples**2, net_order2_terms, dim_max))
        
        
        podk_data_order1_omega12 = np.zeros((grid_points, discretisation_samples, dim_max, dim_max))
        
        
                  
        #precautinary step
        for a in range(dim_max):
            
                podk_data_order2_omega12[:,:,:,a] = podk_data_order2_omega1
            
                podk_data_order1_omega12[:,:,:, a] = podk_data_order1_omega1
            
                          
                order2, dim1, dim2 = -1, 1, 2
            
                while dim1 < dim_max:
                    dim2  = 2 
                    while dim2 <= dim_max:
                        if dim2 <= dim1:
                            pass
                
                        else:
                            
                            order2 += 1
                            
                         
                            if dim1-1 == a:
                                    
                                discretisation_order2_omega12 = np.stack((discretisation[:,1,dim1-1], discretisation[:,0, dim2-1]), axis = 1)
                                
                                podk_data_order1_omega12[:,:,dim1-1, a] = podk_data_order1_omega2[:,:,dim1-1]
                                
                                Y = data_order2[:,:,order2]
                         
                                X = Quad_points_order2[:,:,order2]
                                
                                index1 = order_2_index[:,:,order2]
                                
                                podk_data_order2_omega12[:,:,order2, a], Quad_discard, index_discard = krig.gaussion_process_regression.decomposition_2D(Y, X, index1, discretisation_order2_omega12, Kernel, theta, discretisation_samples, False)
                                
                                
                            if dim2-1 == a:
                                
                                discretisation_order2_omega12 = np.stack((discretisation[:,0, dim1-1], discretisation[:,1, dim2-1]), axis = 1)
                                
                                podk_data_order1_omega12[:,:,dim2-1, a] = podk_data_order1_omega2[:,:,dim2-1]
                                
                                Y = data_order2[:,:,order2]
                         
                                X = Quad_points_order2[:,:,order2]
                                
                                index1 = order_2_index[:,:,order2]
                                
                                podk_data_order2_omega12[:,:,order2, a], Quad_discard, index_discard = krig.gaussion_process_regression.decomposition_2D(Y, X, index1, discretisation_order2_omega12, Kernel, theta, discretisation_samples, False)
                            
                                            
                        dim2 += 1
                    dim1 += 1
                   
        
        
        
        
        
        
        
        
        #Call necceary anova results
        
        X_omega1, index_discard = an.cANOVA_decomposition.select(f0_data, podk_data_order1_omega1, podk_data_order2_omega1, trunc_order, ignore_order2_index)
        
        X_omega2, index_discard = an.cANOVA_decomposition.select(f0_data, podk_data_order1_omega2, podk_data_order2_omega2, trunc_order, ignore_order2_index)
        
        
        X_omega12 = np.zeros((grid_points, discretisation_samples**dim_max, dim_max))
        
        for i in range(dim_max):           
            
            
            X_omega12[:,:,i], index_discard = an.cANOVA_decomposition.select(f0_data, podk_data_order1_omega12[:,:,:,i], podk_data_order2_omega12[:,:,:,i], trunc_order, ignore_order2_index)
            
        
        
        #compute mean
        
        mean = np.zeros((grid_points))
        
        
        for z in range(grid_points):
            
            sumer = 0
        
            for i in range(discretisation_samples**dim_max):
            
                sumer = sumer + X_omega1[z,i]
                
            mean[z] = sumer / discretisation_samples**dim_max
        
        
        
        #compute variance
        
        var = np.zeros((grid_points))
        
        for z in range(grid_points):
            
            sumer = 0
        
            for i in range(discretisation_samples**dim_max):
            
                sumer = sumer + (X_omega1[z,i] - mean[z])**2
                
            var[z] = sumer / discretisation_samples**dim_max
        
        
        
        #compute partial variance
        
        var_partial = np.zeros((grid_points, dim_max))
        
        for a in range(dim_max):
        
            for z in range(grid_points):
                
                sumer = 0
            
                for i in range(discretisation_samples**dim_max):
                
                    sumer = sumer + (X_omega2[z,i])*(X_omega12[z,i,a] - X_omega1[z,i])
                    
                var_partial[z, a] = sumer / discretisation_samples**dim_max
        
        
        
        
        
        #compute sobol index
        
        S_j = np.zeros((grid_points, dim_max))
        
        for i in range(dim_max):
            
            for z in range(grid_points):
            
                S_j[z, i] = var_partial[z, i] / var[z]
        
        
        
        return S_j, var_partial, var, mean
    
    
        
        
        