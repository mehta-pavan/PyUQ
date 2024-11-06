#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 17:47:14 2018

@author: pmehta
"""

import numpy as np
import anova_adaptive_creteria as anct
import anova_decomposition as an
import cross_validation as cv
import krig_for_anova as generate_anova_terms
import reinforce as residual
import pod as pod
import sobol_indices_3 as statsx
        


class capk:
    
    """
    
    c-ANOVA / POD / Kigging main file
    
    """
    
    
    
    
    def intialisation(f0_data, data_order1, data_order2, data_validation, order2_selection_thershold = 0.05, order2_adative_thershold = 0.9):
        
       """
       
       This method is used for intilaising the cAPK alogorithm
       
       """
        
        
        
        #Evalute the order to trncation required
        
        relative_diff, mean_ev, trunc_order = anct.higher_order_selection.order_2_selection(f0_data, data_order1, data_order2, data_validation, order2_selection_thershold)
        
        if trunc_order == 2:
            #implement adaptive anova
            
            active_dim, var_i, var_ij, net_var_order1, var, ignore_order2_index_var = anct.adpative_anova.var_creteria_order2_terms(f0_data, data_order1, data_order2, None, order2_adative_thershold, False, trunc_order)
            
        else:
            ignore_order2_index_var = None
            
        #Perform ANOVA decompoisition
        
        f_anova, index_intial = an.cANOVA_decomposition.select(f0_data, data_order1, data_order2, trunc_order, ignore_order2_index_var)
        
        
        return f_anova, trunc_order, ignore_order2_index_var, index_intial
    
    
    def interate_1D(data_order1, Quad_points, lambda_CV1 = 0.05, domain_percentage_convergence = 0.95, discrestisation = None, Kernel = "poly_cubic_spline", theta = 0.1, disceretistation_samples = 20, auto_uniform_sampling = True,  domain_filter = False, filter_thershohld = None, active_domain = False, active_domain_threshold = None,  L1_sense = True, L2_sense = False):
        
        """
        
        Adding sample in hyperlines
        
        
        """
        
        
        
        
        #krigging - not required for intitalisation
        
        podk_data_order1, podk_Quad_order1, podk_index_order1 = generate_anova_terms.built_data.generating_order1_terms(data_order1, Quad_points, discrestisation, Kernel, theta, disceretistation_samples)
        
        
        #check convergence       
        
        CV_score_with_out_filteration_order1, CV_score_filter_domain_order1, Domain_not_converged_order1 = cv.cross_validation.fit_1D(podk_data_order1, data_order1, Quad_points, lambda_CV1, domain_percentage_convergence, discrestisation, Kernel, theta, auto_uniform_sampling,  domain_filter, filter_thershohld, active_domain, active_domain_threshold,  L1_sense, L2_sense)
        
        
        return podk_data_order1, podk_Quad_order1, podk_index_order1, CV_score_with_out_filteration_order1, CV_score_filter_domain_order1, Domain_not_converged_order1 
    
    
    def interate_2D(f0_data, data_order1, data_order2, Quad_points_order2, order_2_index, lambda_CV2 = 0.05, domain_percentage_convergence = 0.95, discrestisation = None, Kernel = "poly_cubic_spline", theta = 0.1, disceretistation_samples = 20, auto_uniform_sampling = True, domain_filter = False, filter_thershohld = None, active_domain = False, active_domain_threshold = None, active_coupling = False, active_coupling_thershold = None,  L1_sense = True, L2_sense = False):
        
             
        """
        
        Adding samples in hyperplanes
        
        
        """
        
        
        
        #krigging - not required for intitalisation
        podk_data_order2, podk_Quad_order2, podk_index_order2 = generate_anova_terms.built_data.generating_order2_terms(data_order2, Quad_points_order2, order_2_index, discrestisation, Kernel, theta, disceretistation_samples, auto_uniform_sampling)
        
        
        #check convergence 
        
        CV_score_with_out_filteration_order2, CV_score_filter_domain_order2, Domain_not_converged_order2 = cv.cross_validation.fit_2D(podk_data_order2,f0_data, data_order1, data_order2, Quad_points_order2, order_2_index, lambda_CV2, domain_percentage_convergence, discrestisation, Kernel, theta, auto_uniform_sampling, domain_filter, filter_thershohld, active_coupling, active_coupling_thershold, active_domain, active_domain_threshold,  L1_sense, L2_sense)
        
        
        return podk_data_order2, podk_Quad_order2, podk_index_order2, CV_score_with_out_filteration_order2, CV_score_filter_domain_order2, Domain_not_converged_order2
    
    
    
    def response_surface(f0_data, data_order1, data_order2, data_validation, f_anova_corners, Quad_points, Quad_index, residual_modelling = True, trunc_order = 2, ignore_order2_index_var = None, POD_decomposition_1D = False, POD_decomposition_2D = False, pod_treshold = 0.999, auto_uniform_sampling = True, discrestisation = None,  Kernel = "poly_cubic_spline", theta = 0.1):
        
        
        """
        
        Build response surface for 4 dimeniosnal functions
        
        """
        
        
               
        #perform POD for order 1 terms
        
          
        if (POD_decomposition_1D == True):
            
            shape = data_order2.shape
            grid_points, computational_points, net_order1_terms = shape[0], shape[1], shape[2]
            
            podk_data_order1 = np.zeros((grid_points, computational_points, net_order1_terms))
            
            for i in range(net_order1_terms):
                
                podk_data_order1[:,:,i], coeff_v_eig_vector_ignore, Phi_ignore = pod.POD.pod(data_order1[:,:,i], pod_treshold)
        else:
            podk_data_order1 = data_order1
        
        
        
        
        #perform POD for order 2 terms
        
        
        if (POD_decomposition_2D == True):
            
            shape = data_order1.shape
            grid_points, computational_points, net_order2_terms = shape[0], shape[1], shape[2]
            
            podk_data_order2 = np.zeros((grid_points, computational_points, net_order2_terms))
            
            for i in range(net_order2_terms):
                
                podk_data_order2[:,:,i], coeff_v_eig_vector_ignore, Phi_ignore = pod.POD.pod(data_order2[:,:,i], pod_treshold)
        else:
            podk_data_order2 = data_order2
        
        
        
        #Perform ANOVA decompoisition
        
        f_anova, index_anova = an.cANOVA_decomposition.select(f0_data, podk_data_order1, podk_data_order2, trunc_order, ignore_order2_index_var)
        
        
        
        #residual modelling
        if residual_modelling == True:
            
            shape = f_anova.shape
            computational_points = shape[1]
            
            #for problems in 4 dimennsion - change the value accordingly
            res_disceretistation_samples = computational_points**(1/4)
        
            f_reinforced, index_res_krig, Quad_res_krig = residual.reinforce.residual_4D(f0_data, podk_data_order1, podk_data_order2, data_validation, f_anova_corners, Quad_points, Quad_index, trunc_order, ignore_order2_index_var, discrestisation, auto_uniform_sampling, res_disceretistation_samples, Kernel, theta)
        
        else:
            
            f_reinforced = f_anova
            index_res_krig = index_anova
        
        
        return f_reinforced, index_res_krig
    
    
    
    
    def capk_statistics(f0_data, data_order1, data_order2, Quad_data_order1, Quad_points_order2, order_2_index, trunc_order = 2, ignore_order2_index = None, Kernel = "poly_cubic_spline", theta = 0.1, discretisation_samples = 20):
        
        """
        
        This method performs Quasi Monte Carlo using Sobol sequence in 4 dimensions
        
        """
        
        
        
        #compute sobol indices       
        
        S_j, var_partial, var, mean = statsx.sobol_indices.indices(f0_data, data_order1, data_order2, Quad_data_order1, Quad_points_order2, order_2_index, trunc_order, ignore_order2_index, Kernel, theta, discretisation_samples)
        
        return S_j, var_partial, var, mean
        
    
    
    
    
        