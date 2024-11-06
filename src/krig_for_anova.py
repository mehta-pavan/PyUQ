"""
Author: Pavan Mehta
Email: mehtapavanp@gmail.com

"""


import numpy as np
import krigging as krig

class built_data:
    
    """
    This file performs 1D and 2D krigging with bulding the matrix as per required by the ANOVA files
  
       
    
    """
    
    
    
    def generating_order1_terms(data_order1, Quad_points, discrestisation = None, Kernel = "poly_cubic_spline", theta = 0.1, disceretistation_samples = 20, auto_unifrom_sampling = True):
        
        
        
        
        
        shape = Quad_points.shape
        
        net_Q_points_per_dim, dim_max = shape[0], shape[1]
      
        
        
        shape1 = data_order1.shape
        
        grid_points = shape1[0]
               
        Quad_1 = np.zeros(((disceretistation_samples), dim_max))
        
        index1 = np.zeros(((disceretistation_samples), dim_max), dtype = int)
        
        mu_all = np.zeros((grid_points, disceretistation_samples, dim_max))
        
        for i in range(dim_max):
            
                    
                    X = np.zeros((net_Q_points_per_dim))    
        
                            
                    Y = np.zeros((net_Q_points_per_dim))
                    
                   # mu_1 = np.zeros((grid_points, disceretistation_samples))                   
                                         
                                        
                    for l in range(grid_points): 
                         
                        Y = data_order1[l,:,i]
                         
                        X = Quad_points[:,i]
                        
                        if auto_unifrom_sampling == True:
                            
                            discrestisation_1 = None
                        else:
                            discrestisation_1 = discrestisation[:,i]
                            
                                                                         
                        discrestisation1, mu, variance = krig.gaussion_process_regression.fit_1D(X,Y,discrestisation_1 , Kernel, theta, disceretistation_samples)
                         
                        mu_all[l, :, i] = mu.reshape(-1)
                        
                        
                    
                   # mu_all[:,:,i] = mu_1[:,:]
                    
                    for l in range(disceretistation_samples):
                        
                        Quad_1[l, i] = np.round((discrestisation1[l]), decimals = 3)
                        
                        index1[l, i] = l
                                           
                       
                        
       # mu_all = mu_all[:,disceretistation_samples:]
                                                      
                
        return mu_all, Quad_1, index1
    
    
    
    
    
    
    def generating_order2_terms(data_order2, Quad_points_order2, order_2_index, discrestisation = None, Kernel = "poly_cubic_spline", theta = 0.1, disceretistation_samples = 20, auto_uniform_sampling = True):
        
        
        shape = data_order2.shape
        
        grid_points, computational_points_order2, net_order2_terms = shape[0], shape[1], shape[2]
        
                
        #mu_all, Quad, index = np.zeros((grid_points, disceretistation_samples**2, net_order2_terms)) , np.zeros((disceretistation_samples**2, 2, net_order2_terms)) , np.zeros((disceretistation_samples**2, 2, net_order2_terms))
        
        mu_all = np.zeros((grid_points, disceretistation_samples**2, net_order2_terms))
        
        Quad = np.zeros(((disceretistation_samples), 2, net_order2_terms))
        
        index = np.zeros((disceretistation_samples**2, 2, net_order2_terms))
        
        for i in range(net_order2_terms):
            
                            
                Y = data_order2[:,:,i]
                         
                X = Quad_points_order2[:,:,i]
                
                index1 = order_2_index[:,:,i]
                
                 
                if auto_uniform_sampling == True:
                            
                    discrestisation_1 = None
                else:
                    discrestisation_1 = discrestisation[:,:,i]
                
                mu_all[:,:, i], Quad[:,:, i], index[:,:, i] = krig.gaussion_process_regression.decomposition_2D(Y, X, index1,discrestisation_1, Kernel, theta, disceretistation_samples, auto_uniform_sampling)
        
        
        
        return mu_all,  Quad, index
        