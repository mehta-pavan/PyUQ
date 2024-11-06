#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 18:10:56 2018

@author: pmehta
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: pmehta

Email: mehtapavanp@gmail.com
"""

import numpy as np
import krigging as krig
import stat_var_expect as stat


class CV:
    
    """
    This class gives the different Cross Validation creteria
    
    Availabe creteria:
        1. L2 sense
        2. L1 sense
        3. L1 pointwise
    
    
    """
    
    
    
    def global_l2(train_data, test_data, f0_data = None, data_order1 = None, data_order2 = None, get_domain_filter = False, filter_thershohld = None, get_active_domain = False, active_domain_threshold = None, get_active_coupling = False, active_coupling_thershold = None):
        
        """
        This funtions perfroms the Cross vaildation in an L2 sense 
        
        """
        
         
        if get_domain_filter == True:
                test_data_filtered = domain_filter.fit(test_data, filter_thershohld)
                train_data_filtered = domain_filter.fit(train_data, filter_thershohld)
        else:
                test_data_filtered = test_data
                train_data_filtered = train_data
            
            
        if get_active_domain == True:
            test_data_filtered = masking.active_domain(test_data_filtered, active_domain_threshold)
            train_data_filtered = masking.active_domain(train_data_filtered, active_domain_threshold)
                
        else:
            test_data_filtered = test_data
            train_data_filtered = train_data_filtered
        
        
        
        if get_active_coupling == True:
            
            test_data_filtered = masking.active_coupling(test_data_filtered, f0_data, data_order1, data_order2, active_coupling_thershold, None, False)
            train_data_filtered = train_data = masking.active_coupling(train_data_filtered, f0_data, data_order1, data_order2, active_coupling_thershold, None, False)
        else:
            test_data_filtered = test_data
            train_data_filtered = train_data
        
        
        shape = test_data.shape
        
        grid_points, net_computational_data = shape[0], shape[1]
        
        CV_score = np.zeros((grid_points))
        
        for z in range(grid_points):
            
            sumer, sumer1 = 0, 0
            
            for i in range(net_computational_data):
                
                sumer = sumer + (train_data_filtered[z,i] - test_data_filtered[z,i])**2 
                
                sumer1 = sumer1 + (train_data[z,i])**2
                
            CV_score[z] = np.sqrt(sumer) / np.sqrt(sumer1)
        
        return CV_score
    
    
    
        
    def pointwise_l1(train_data, test_data, f0_data = None, data_order1 = None, data_order2 = None, get_domain_filter = False, filter_thershohld = None, get_active_domain = False, active_domain_threshold = None, get_active_coupling = False, active_coupling_thershold = None):
        
        
        """
        This funtions perfroms the Cross vaildation in an L1 pointwise 
        
        """
         
        if get_domain_filter == True:
                test_data_filtered = domain_filter.fit(test_data, filter_thershohld)
                train_data_filtered = domain_filter.fit(train_data, filter_thershohld)
        else:
                test_data_filtered = test_data
                train_data_filtered = train_data
            
            
        if get_active_domain == True:
            test_data_filtered = masking.active_domain(test_data_filtered, active_domain_threshold)
            train_data_filtered = masking.active_domain(train_data_filtered, active_domain_threshold)
                
        else:
            test_data_filtered = test_data
            train_data_filtered = train_data_filtered
        
        
        
        if get_active_coupling == True:
            
            test_data_filtered = masking.active_coupling(test_data_filtered, f0_data, data_order1, data_order2, active_coupling_thershold, None, False)
            train_data_filtered = train_data = masking.active_coupling(train_data_filtered, f0_data, data_order1, data_order2, active_coupling_thershold, None, False)
        else:
            test_data_filtered = test_data
            train_data_filtered = train_data
        
        
        CV_score = np.absolute(train_data_filtered - test_data_filtered) / np.absolute(train_data_filtered)
        
        return CV_score
    
    
    
    
    
    def global_l1(train_data, test_data, f0_data = None, data_order1 = None, data_order2 = None, get_domain_filter = False, filter_thershohld = None, get_active_domain = False, active_domain_threshold = None, get_active_coupling = False, active_coupling_thershold = None):
        
        
        """
        This funtions perfroms the Cross vaildation in an L1 sense 
        
        """
       
        
        if get_domain_filter == True:
                test_data_filtered = domain_filter.fit(test_data, filter_thershohld)
                train_data_filtered = domain_filter.fit(train_data, filter_thershohld)
        else:
                test_data_filtered = test_data
                train_data_filtered = train_data
            
            
        if get_active_domain == True:
            test_data_filtered = masking.active_domain(test_data_filtered, active_domain_threshold)
            train_data_filtered = masking.active_domain(train_data_filtered, active_domain_threshold)
                
        else:
            test_data_filtered = test_data
            train_data_filtered = train_data_filtered
        
        
        
        if get_active_coupling == True:
            
            test_data_filtered = masking.active_coupling(test_data_filtered, f0_data, data_order1, data_order2, active_coupling_thershold, None, False)
            train_data_filtered = train_data = masking.active_coupling(train_data_filtered, f0_data, data_order1, data_order2, active_coupling_thershold, None, False)
        else:
            test_data_filtered = test_data
            train_data_filtered = train_data
        
       
        
        shape = test_data.shape
        
        grid_points, net_computational_data = shape[0], shape[1]
        
        CV_score = np.zeros((grid_points))
        
        for z in range(grid_points):
            
            sumer, sumer1 = 0, 0
            
            for i in range(net_computational_data):
        
                sumer = sumer + np.absolute(train_data_filtered[z,i] - test_data_filtered[z,i])
    
                sumer1 = sumer1 + np.absolute(train_data[z,i])
                
            CV_score[z] = sumer / sumer1
        
        return CV_score
    
    
    
#--------------------------------------------------------------------------------------------------------------------------------------



    
class masking:
    
    """
    This class is reserved for masking the domain for convergeence creteria's as oultined in Magri and Saguat (2016) artilce on c-APK
    
    """
    
    
    def active_domain(f_data, active_domain_threshold):
        
        """
        This method idetifies the active domin. In c-APK terminology an acitve domina is a part of domain above a given thershold, 
        which is used to compute the global convergence 
        
        """
        
        shape = f_data.shape
        
        grid_points = shape[0]
        computational_points = shape[1]
        #net_terms = shape[2]
        
        #active_dom = np.zeros((grid_points, computational_points, net_terms))
        active_dom = np.zeros((grid_points, computational_points))
        
        #for k in range(net_terms):
        
        for i in range(computational_points):
                for j in range(grid_points):
                    
                    data_point = f_data[j,i]
                    
                    if (data_point >= active_domain_threshold):
                        
                        active_dom[j,i] = data_point
                    
                    else:
                        
                        active_dom[j,i] = 1e-10
                    
        
        return active_dom
        
        
    
    
    
    def active_coupling(f_data, f0_data, data_order1, data_order2, active_coupling_thershold, active_domain_threshold = None, get_active_domain_auto = True):
        
        
        """
        
        This method is reserved for seleclting active second order couplings using parital varinces 
        
        
        """
        
        #get active domain data
        if get_active_domain_auto == True:
            
            
            active_dom = masking.active_domain(f_data, active_domain_threshold)
            
        else:
            
            active_dom = f_data
            
        
      
                
        #call ANOVA Order 2 terms variances
        
        variance_order2 = stat.variance.order2_terms(f0_data, data_order1, data_order2)
        
        
        #Loop initilisation parameters
        
        shape = f_data.shape
        
        grid_points = shape[0]
        compuatational_points = shape[1]
        
        shape = variance_order2.shape
        
        net_terms = shape[1]
        
        active_coupling_mask = np.zeros((grid_points, compuatational_points, net_terms))
        
        #therschold for active coupling mask
        
        for i in range(grid_points):
            
            for j in range(net_terms):
                
                for k in range(compuatational_points):
                
                
        
                    if active_dom[i,k] == 1e-10:
                                           
                        variance_order2[i,j] = 1e-10
                        
                    else:
                        pass
                                    
                    
                    data_point = variance_order2[i,j]
                    
                                    
                    if (data_point >= active_coupling_thershold):
                        
                        active_coupling_mask[i,k] = data_point
                        
                    else:
                        
                        active_coupling_mask[i,k] = 1e-10
                        
        
        index = np.where((active_coupling_mask != 1e-10))
        
        
        return active_coupling_mask, index



#--------------------------------------------------------------------------------------------------------------------------------------


        
            
class domain_filter:
    
    def fit(f_data, filter_thershohld):
        
        """
        Filtering the domain, values above the thershlod only retainted.
        
        """
        
        
        shape = f_data.shape
        
        grid_points, computational_points = shape[0], shape[1]
        
        
        #for k in range(net_terms):
        
        for j in range(computational_points):
                
                for i in range(grid_points):
                    
                    
                    data_point = f_data[i,j]
                    
                    if data_point > filter_thershohld:
                        
                        pass
                    
                    else:
                    
                        f_data[i,j] = 1e-10
                    
        
        return f_data
    



#--------------------------------------------------------------------------------------------------------------------------------------




class domain_percentage_convergence:
    
    
    def fit(CV_score, lamda_CV, domain_percentage_convergence = 0.95):
        
        """
        
        This method computes the percentage of domain not connverged
        
        """
        
        #CV_score = CV.pointwise_l1(train_data, test_data)
        
        shape = CV_score.shape
        
        grid_points, computational_points, net_terms = shape[0], shape[1], shape[2]
        
        Domain_not_converged = np.zeros((computational_points, net_terms))
        
        for k in range(net_terms):
        
            for j in range(computational_points):
                
                counter = 0
                
                for i in range(grid_points):
                    
                    data_point = CV_score[i,j,k]
                    
                    if data_point > lamda_CV:
                        
                        counter += 1
                        
                    else:
                        
                        pass
                    
                Domain_not_converged[j,k] = counter/grid_points
         
            
        
                
        if np.all((Domain_not_converged < domain_percentage_convergence)):
                    
            print("Stop Iteration on account of reaching set domain convergence")
        else:
                    
            print("CONTINUE Iteration on account of reaching set domain NOT converged")
                
        
        return Domain_not_converged
        
    
                
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
