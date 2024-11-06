#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 18:27:50 2018

@author: pmehta
"""

class cANOVA_decomposition:
    
    
    def select(f0_data, data_order1, data_order2, trunc_order = 2, ignore_order2_index = None):
        
        
        shape = data_order1.shape
        
        dim_max = shape[2]
        
        
        # c-ANOVA Decoposition selection
        
        
        if (dim_max == 4):
            
            import anova_4D as an
            
            if (trunc_order == 1):
                
                
                #Call ANOVA decomposition in 4 Dimenisons with trunc_order 1
                f_anova, index = an.cANOVA.decomposition_4D_v1(f0_data, data_order1)
             
                
            elif (trunc_order == 2):
                
                #Call ANOVA decomposition in 4 Dimenisons with trunc_order 2
                f_anova, index = an.cANOVA.decomposition_4D_v2(f0_data, data_order1, data_order2, ignore_order2_index)
            
            else:
                
                raise "Only tencation order 1 and 2 permitted"
         
        else:
            
            raise "Only 4 or 8 Dimensional functions permitted"
        
        
        
        
        return f_anova, index
