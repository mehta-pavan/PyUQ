#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 15:15:24 2018

@author: pmehta
"""

import numpy as np
import scipy as sp
from scipy import linalg
from matplotlib import pyplot as plt
"""
    This file preform POD in N Dimensions

"""

class POD:
    
    def pod(f_data_matrix, pod_treshold = 0.99):
        
        shape = f_data_matrix.shape
        
        grid_points, snapshots = shape[0], shape[1]
        
        Snapshot_matrix = np.dot(f_data_matrix, ((np.eye((snapshots))) - ((1/snapshots)*(np.ones((snapshots, snapshots))))))
        
        
        #SVD Decompisition 
        
        U,sig,Vt = linalg.svd(Snapshot_matrix)
        
        
        #Private Variables intitialzing
        explained_var = trunc_order = sumer = 0
        
        
        

        while (explained_var <= pod_treshold):
            sumer += sig[trunc_order]
            explained_var =  sumer/sig.sum()
            trunc_order += 1


        r = trunc_order - 1 #looping effects - code needs imporvement
        

        # Performing Truncation - SVD
        
        if (r == 0):
            sig_r = np.zeros((1))
        
            sig_r[r] = sig[r]
        
        else:
    
            sig_r = np.zeros((trunc_order,trunc_order))
            
            for i in range(trunc_order):
                sig_r[i,i] = sig[i]


        U_r = np.zeros((grid_points,trunc_order))
        

        U_r = U[:,:trunc_order]
        


        #Buliding POD basis vectors

        Phi = np.dot(U_r, sig_r)
        
        #Phi = Phi[:,0]



        #Computing coeff. for K-L Expansion
        
        #------------------------------------------------------------------------

        # Correaltion matrix

        Corr_matrix = (1/snapshots)*(np.dot(Snapshot_matrix.transpose(), Snapshot_matrix))
        
                
        #Finding the right eigen vectors for any general matrix
        #coeff_v_eig_val, coeff_v_eig_vector = sp.linalg.eig(Corr_matrix, right=True)
        
        
        #Finding the right eigen vectors for an Hermitian or Symmetric matrix
        coeff_v_eig_val, coeff_v_eig_vector = sp.linalg.eigh(Corr_matrix)
        
        
        #Arraning the vectors as per deceresing egine values
        idx = np.arange(snapshots-1, -1, -1)
        
        coeff_v_eig_vector = coeff_v_eig_vector[:, idx]
        
        #coeff_v_eig_vector = coeff_v_eig_vector[idx, :]
        
        


        #Computing Coeff.

        #coeff_v = coeff_v_eig_vector[0,:]

        #KL Expansion

        result = np.zeros((grid_points, snapshots))



        for j in range(snapshots):
            for i in range(grid_points):
                sumer = 0
                
                if (r == 0):
                    
                                      
                    result[i,j] = (coeff_v_eig_vector[j,0]*Phi[i])
                     
                    
                else:
                    
                    for k in range(trunc_order):
                        sumer = sumer + (coeff_v_eig_vector[j,k]*Phi[i,k])
                    result[i,j] = sumer
        
        
        
        return result, coeff_v_eig_vector, Phi
