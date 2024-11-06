#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:22:53 2018

@author: pmehta
"""

import sin_test as test_func
import capk_v2 as capk


class testx:
    
    """
    
    This scipt demonstares the use of CAPK alogithm
    
    """ 
    
    
    
    def test1(T0 = 300, A1 = 10, A2 = 10, A3 = 10, A4 = 10, net_quad_point = 4, A1_range = 10,  A2_range = 10, A3_range = 10 ,  A4_range = 10, grid_poins = 100, trunc_order = 2, ignore_order2_index = None, Kernel = "poly_cubic_spline", theta = 0.1, discretisation_samples = 20):
        
        """
        
        This method demonsstrets the use of cAPK algorithm
        
        """
        
        #call the case
        f0, Quad_points, f1, order1_index, f2, Quad_points_order2, order2_index, data_validation, Quad_points_validation = test_func.sin_test.test(T0, A1, A2, A3, A4, net_quad_point, A1_range,  A2_range, A3_range,  A4_range, grid_poins)
        
        
        #get intialinsation data
        f_anova, trunc_order, ignore_order2_index_var, index_intial = testx.intiail(T0, A1, A2, A3, A4, A1_range,  A2_range, A3_range,  A4_range, grid_poins)
        
        print("step 1 done")
        
        #itertae over hyperlines
        podk_data_order1, podk_Quad_order1, podk_index_order1, CV_score_with_out_filteration_order1, CV_score_filter_domain_order1, Domain_not_converged_order1  = capk.capk.interate_1D(f1, Quad_points, lambda_CV1 = 0.05, domain_percentage_convergence = 0.95, discrestisation = None, Kernel = "poly_cubic_spline", theta = 0.1, disceretistation_samples = discretisation_samples, auto_uniform_sampling = True,  domain_filter = False, filter_thershohld = None, active_domain = False, active_domain_threshold = None,  L1_sense = True, L2_sense = False)
        
        print("step 2 done")
        
        #iterate over hyperplanes
        podk_data_order2, podk_Quad_order2, podk_index_order2, CV_score_with_out_filteration_order2, CV_score_filter_domain_order2, Domain_not_converged_order2 = capk.capk.interate_2D(f0, f1, f2, Quad_points_order2, order2_index, lambda_CV2 = 0.05, domain_percentage_convergence = 0.95, discrestisation = None, Kernel = "poly_cubic_spline", theta = 0.1, disceretistation_samples = discretisation_samples, auto_uniform_sampling = True, domain_filter = False, filter_thershohld = None, active_domain = False, active_domain_threshold = None, active_coupling = False, active_coupling_thershold = None,  L1_sense = True, L2_sense = False)
        
        print("step 3 done")

        #model residual and build the response surface
        f_reinforced, index_res_krig = capk.capk.response_surface(f0, podk_data_order1, podk_data_order2, data_validation, f_anova, Quad_points_validation, index_intial, residual_modelling = True, trunc_order = 2, ignore_order2_index_var = None, POD_decomposition_1D = False, POD_decomposition_2D = False, pod_treshold = 0.999, auto_uniform_sampling = True, discrestisation = None,  Kernel = "poly_cubic_spline", theta = 0.1)
        print("step 4 done")
        
        
        #compute stats
        
        S_j, var_partial, var, mean = capk.capk.capk_statistics(f0, podk_data_order1, podk_data_order2, podk_Quad_order1, podk_Quad_order2, podk_index_order2, trunc_order, ignore_order2_index, Kernel, theta, discretisation_samples*2)
        print("step 5 done")
        
        return podk_Quad_order1, podk_index_order1, podk_Quad_order2, podk_index_order2
    
    def intiail(T0 = 300, A1 = 10, A2 = 10, A3 = 10, A4 = 10, A1_range = 10,  A2_range = 10, A3_range = 10 ,  A4_range = 10, grid_poins = 100):
        
        """
        This method is used strictly for intilaistion only - data to be supplied accordinly
        
        """
        
        
        #call the case data strictly for intilisation only
        f0, Quad_points, f1, order1_index, f2, Quad_points_order2, order2_index, data_validation, Quad_points_validation = test_func.sin_test.test(T0, A1, A2, A3, A4, 2, A1_range,  A2_range, A3_range,  A4_range, grid_poins)
        
        #compute the necessary data
        f_anova, trunc_order, ignore_order2_index_var, index_intial = capk.capk.intialisation(f0, f1, f2, data_validation, order2_selection_thershold = 0.05, order2_adative_thershold = 0.9)
         
        return f_anova, trunc_order, ignore_order2_index_var, index_intial
    
  
        
        
