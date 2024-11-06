#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 14:04:14 2018

@author: pmehta

Email: mehtapavanp@gmail.com
"""

import numpy as np
import loo as loo
import cv as cv

class cross_validation:
    
    """
    
    This class tests the convergence accoring to Leave one out alorithm with L1, L2 and other creteria as per the artilce by Luca and Sagaut(2016) on cAPK
    
    """
    
    
    def fit_1D(data_matrix, data_order1, Quad_points, lambda_CV1 = 0.05, domain_percentage_convergence = 0.95, discrestisation = None, Kernel = "poly_cubic_spline", theta = 0.1, auto_uniform_sampling = True,  domain_filter = False, filter_thershohld = None, active_domain = False, active_domain_threshold = None,  L1_sense = True, L2_sense = False):
        
        """
        
        This method is only for 1 dimenionsnal data
        
        """
        
        
        shape = data_matrix.shape
        
        grid_points, net_computational_points, dim_max = shape[0], shape[1], shape[2]
        
        shape = Quad_points.shape
        
        net_quad_points = shape[0]
        
        CV_score_filter_domain = np.zeros((grid_points, net_quad_points - 2, dim_max))
        
        CV_score_with_out_filteration, test_data = loo.leave_one_out.fit_1D(data_matrix, data_order1, Quad_points, discrestisation, Kernel, theta, auto_uniform_sampling,  L1_sense, L2_sense)
        
        
       
        
        
        for i in range(dim_max):
            
                        
            
            counter = 0
            
            for j in range(net_quad_points):
                
                 
                if ((j == 0) or (j == (net_quad_points -1))):
                    pass
                else:
                
                    train_data = data_matrix[:,:,i]
                    
                    if L1_sense == True:
                    
                        CV_score_filter_domain[:,counter,i] = cv.CV.global_l1(train_data, test_data[:,:,counter,i], None, None, None, domain_filter, filter_thershohld, active_domain, active_domain_threshold, False, None)
                        counter += 1
                    elif L2_sense == True:
                        CV_score_filter_domain[:,counter,i] = cv.CV.global_l2(train_data, test_data[:,:,counter,i], None, None, None, domain_filter, filter_thershohld, active_domain, active_domain_threshold, False, None)
                        counter += 1
                    else:
                        raise "Cross validation criteria: L1_sense or L2_sense must be True"
                    
                
        Domain_not_converged = cv.domain_percentage_convergence.fit(CV_score_filter_domain, lambda_CV1, domain_percentage_convergence)
        
        
        return CV_score_with_out_filteration, CV_score_filter_domain, Domain_not_converged
    
    
    
    #-----------------------------------------------------------------------------------------------------------------------------------------------
    
    
    
    def fit_2D(data_matrix,f0_data, data_order1, data_order2, Quad_points, index, lambda_CV2 = 0.05, domain_percentage_convergence = 0.95, discrestisation = None, Kernel = "poly_cubic_spline", theta = 0.1, auto_uniform_sampling = True, domain_filter = False, filter_thershohld = None, active_coupling = False, active_coupling_thershold = None, active_domain = False, active_domain_threshold = None,  L1_sense = True, L2_sense = False):
        
        
        """
        
        This method is only for 2 dimenisional functions
        
        """
        
        
        shape = Quad_points.shape
        
        net_Q_points_per_dim = shape[0]
               
        shape1 = data_matrix.shape
        
        grid_points, computational_points, net_order2_terms = shape1[0], shape1[1], shape1[2]
          
        
              
        CV_score_filter_domain = np.zeros((grid_points, net_Q_points_per_dim**2 - 4, net_order2_terms))
        
        CV_score_with_out_filteration, test_data = loo.leave_one_out.fit_2D(data_matrix, data_order2, Quad_points, index, discrestisation, Kernel, theta, auto_uniform_sampling, L1_sense, L2_sense)
        
         
               
        
        
        for n in range(net_order2_terms):
            
                        
            
            train_data = data_matrix[:,:,n]
            
           
            
            for m in range(int(net_Q_points_per_dim**2 - 4)):
                
                 
                if L1_sense == True:
                    
                    CV_score_filter_domain[:,m,n] = cv.CV.global_l1(train_data, test_data[:,:,m,n], f0_data, data_order1, data_order2, domain_filter, filter_thershohld, active_domain, active_domain_threshold, active_coupling, active_coupling_thershold)
                    
                elif L2_sense == True:
                    CV_score_filter_domain[:,m,n] = cv.CV.global_l2(train_data, test_data[:,:,m,n], f0_data, data_order1, data_order2, domain_filter, filter_thershohld, active_domain, active_domain_threshold, active_coupling, active_coupling_thershold)
                    
                else:
                    raise "Cross validation criteria: L1_sense or L2_sense must be True"
                
                
                
                
        Domain_not_converged = cv.domain_percentage_convergence.fit(CV_score_filter_domain, lambda_CV2, domain_percentage_convergence)
                
            
                
        
        
        
        
        return CV_score_with_out_filteration, CV_score_filter_domain, Domain_not_converged
        
        
        
        
        
        
        
        
        
        
        
        
        
        