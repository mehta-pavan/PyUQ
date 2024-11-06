#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 15:18:15 2018

@author: pmehta
"""
import numpy as np
from sobol import i4_sobol, prime_ge
import matplotlib.pyplot as pl
from sklearn.preprocessing import MinMaxScaler

"""
In this file there are two classes, one for providing samples and other perfoming rudimentory tests



"""


class sampling:

    def samples(X, sampling = None, dim_max = 2, n_points = 100):
        
        """ This function selects the sampling technique and discretisites it"""
        
        #For robustness
        X = X.reshape(-1,1)
        
        shape = X.shape
        
                
        #Range of X or dataset
        R_1 = min(X)
        R_2 = max(X)
        
        if (sampling == None) or (sampling == "uniform"):
            
            # points we're going to make predictions at.
            samp = np.linspace(R_1,R_2, num = n_points, endpoint = True).reshape(-1,1)
            
            x = samp
                    
        elif (sampling == "sobol"):
                
            sob1 = np.zeros((n_points,dim_max))
            sampling = np.zeros((n_points, dim_max))
                
            for dim_num in range( 2, dim_max+1):
            
                seed = 0
                qs = prime_ge (dim_num)
                                       
                                        
                for i in range(0, n_points):
                    [ r, seed_out ] = i4_sobol( dim_num, seed )
                    for k in range(dim_num):
                       sob1[i,k] = r[k]
                       
                    out='%6d %6d  '%(seed, seed_out )
                    for j in range (0, dim_num):
                        out+='%10f  '%r[j]
                        #print( out)
                        seed = seed_out   
                
                #scalng samples
                scaling_fac = MinMaxScaler(feature_range=(R_1,R_2))
                scaling_fac = scaling_fac.fit(sob1)
                samp = scaling_fac.transform(sob1)
                
                #for plot or other porposes
                x = np.linspace(R_1,R_2, n_points).reshape(-1,1)
                
            
        elif ((sampling == "sobol") and (dim_max < 2)):
            
            raise "the minimul dimnesion for smapling using sobol sq. is 2: dim_max = 2 or higher"
            
            
        else:
                
                raise "Check Sampling technique permited value : None, uniform, sobol"
        
        
        return samp, x


class test:
    
    def plot_sobol(X, sampling = "sobol", dim_max = 2, n_points = 100):
                       
        samp, x = sampling.samples(X, sampling = "sobol", dim_max = 2, n_points = 100)
        
        Y1, Y2 = samp[:,0], samp[:,1]
        
        #Y1, Y2 = np.sort(Y1), np.sort(Y2)
        
        #Creating a plot
        pl.figure(1)
        pl.clf()
        pl.plot(x, Y1, 'r.', ms=5)
        pl.plot(x, Y2, 'b.', ms = 5)
        pl.savefig('sobol_sequence.png', bbox_inches='tight')
        pl.title('sobol_sequence')
                
        return pl.figure(1)
    
    def plot_uniform(X, sampling = None, dim_max = 2, n_points = 100):
                       
        samp, x = sampling.samples(X, sampling, dim_max, n_points)
        
        Y1 = samp
        
        print("Expented outcome: A Line")
        
        #Creating a plot
        pl.figure(2)
        pl.clf()
        pl.plot(x, Y1, 'r.', ms=5)
        pl.savefig('unifrom-samples.png', bbox_inches='tight')
        pl.title('unifrom-samples, Expented outcome: A Line')
                
        return pl.figure(2)