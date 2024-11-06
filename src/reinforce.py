#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: pmehta
"""

import numpy as np
import anova_decomposition as an
import krigging as krig


class reinforce:
    
    """
    This class provides methods to correct the c_ANOVA results
    
    
    """
       
    
    def residual_4D(U_mean, U_order1, U_order2, f_cfd_corners, f_anova_corners, Quad_points_validation, index_intial, trunc_order = 2, ignore_order2_index = None, discrestisation = None, auto_uniform_sampling = True, disceretistation_samples = 10, Kernel = "poly_cubic_spline", theta = 0.1):
        
        """
        This method coputes the residual values in 4 Dimensions
        
        """
        
        
        #Evaluting values at corner poins
        res_corner_values = f_cfd_corners - f_anova_corners
        
        
        #interpolate X_T using kirgging - residual modelling
        
        X_T, index_res_krig, Quad_res_krig = krig.gaussion_process_regression.decomposition_4D(res_corner_values, Quad_points_validation, index_intial, discrestisation, Kernel, theta, disceretistation_samples, auto_uniform_sampling)
        
        
        
        #calling Anova results
        
        #f_anova, index = an.cANOVA.decomposition_4D_v2(U_mean, U_order1, U_order2)
        
        f_anova, index = an.cANOVA_decomposition.select(U_mean, U_order1, U_order2, trunc_order, ignore_order2_index)
        
        
        #Corren=cted values
        
        f_reinforced = f_anova + X_T
        
        return f_reinforced, index_res_krig, Quad_res_krig