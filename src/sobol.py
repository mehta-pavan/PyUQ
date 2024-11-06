#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 20:18:46 2018

@author: pmehta
"""

import math
from numpy import *
#import numpy as np

from scipy import stats

#*****************************************************************************80
#
## I4_BIT_HI1 returns the position of the high 1 bit base 2 in an integer


def i4_bit_hi1 ( n ):
    i = int ( n )
    bit = 0
    while ( True ):
        if ( i <= 0 ):
            break
        bit += 1
        i = ( i // 2 )
    return bit


#*****************************************************************************80
#
## I4_BIT_LO0 returns the position of the low 0 bit base 2 in an integer.
#
#  Example:



def i4_bit_lo0(n):
    bit = 0
    i = int ( n )
    while ( 1 ):
        bit = bit + 1
        i2 = ( i // 2 )
        if ( i == 2 * i2 ):
            break
        i = i2

    return bit


#*****************************************************************************80
#
## I4_SOBOL_GENERATE generates a Sobol dataset.


def i4_sobol_generate ( m, n, skip ):
    r=zeros((m,n))
    for j in range(1, n+1):
        seed = skip + j - 2
        [ r[0:m,j-1], seed ] = i4_sobol ( m, seed )
    return r




def i4_sobol(dim_num, seed):
    global atmost
    global dim_max
    global dim_num_save
    global initialized
    global lastq
    global log_max
    global maxcol
    global poly
    global recipd
    global seed_save
    global v
    if ( not 'initialized' in globals().keys() ):
        initialized = 0
        dim_num_save = -1

    if ( not initialized or dim_num != dim_num_save ):
        initialized = 1
        dim_max = 40
        dim_num_save = -1
        log_max = 30
        seed_save = -1
    #
        v = zeros((dim_max,log_max))
        v[0:40,0] = transpose([ \
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ])
        v[2:40,1] = transpose([ \
		  1, 3, 1, 3, 1, 3, 3, 1, \
		  3, 1, 3, 1, 3, 1, 1, 3, 1, 3, \
		  1, 3, 1, 3, 3, 1, 3, 1, 3, 1, \
		  3, 1, 1, 3, 1, 3, 1, 3, 1, 3 ])
        v[3:40,2] = transpose([ \
		  7, 5, 1, 3, 3, 7, 5, \
		  5, 7, 7, 1, 3, 3, 7, 5, 1, 1, \
		  5, 3, 3, 1, 7, 5, 1, 3, 3, 7, \
		  5, 1, 1, 5, 7, 7, 5, 1, 3, 3 ])
        v[5:40,3] = transpose([ \
		  1, 7, 9,13,11, \
		  1, 3, 7, 9, 5,13,13,11, 3,15, \
		  5, 3,15, 7, 9,13, 9, 1,11, 7, \
		  5,15, 1,15,11, 5, 3, 1, 7, 9 ])
        v[7:40,4] = transpose([ \
		  9, 3,27, \
		  15,29,21,23,19,11,25, 7,13,17, \
		  1,25,29, 3,31,11, 5,23,27,19, \
		  21, 5, 1,17,13, 7,15, 9,31, 9 ])
        v[13:40,5] = transpose([ \
		  37,33, 7, 5,11,39,63, \
		  27,17,15,23,29, 3,21,13,31,25, \
		  9,49,33,19,29,11,19,27,15,25 ])
        v[19:40,6] = transpose([ \
		  13, \
		  33,115, 41, 79, 17, 29,119, 75, 73,105, \
		  7, 59, 65, 21,	3,113, 61, 89, 45,107 ])
        v[37:40,7] = transpose([ \
		  7, 23, 39 ])
        poly= [ \
		  1,	 3,	 7,	11,	13,	19,	25,	37,	59,	47, \
		  61,	55,	41,	67,	97,	91, 109, 103, 115, 131, \
		  193, 137, 145, 143, 241, 157, 185, 167, 229, 171, \
		  213, 191, 253, 203, 211, 239, 247, 285, 369, 299 ]
        atmost = 2**log_max - 1
#
#	Find the number of bits in ATMOST.
#
        maxcol = i4_bit_hi1 ( atmost )
#
#	Initialize row 1 of V.
#
        v[0,0:maxcol] = 1
    #
    if ( dim_num != dim_num_save ):
#
#	Check parameters.
#
        if ( dim_num < 1 or dim_max < dim_num ):
            print('I4_SOBOL - Fatal error!') 
            print('	The spatial dimension DIM_NUM should satisfy:') 
            print('		1 <= DIM_NUM <= %d'%dim_max)
            print('	But this input value is DIM_NUM = %d'%dim_num)
            return

        dim_num_save = dim_num
#
#	Initialize the remaining rows of V.
#
        for i in range(2 , dim_num+1):
#
#	The bits of the integer POLY(I) gives the form of polynomial I.
#
#	Find the degree of polynomial I from binary encoding.
#
            j = poly[i-1]
            m = 0
            while ( 1 ):
                j = math.floor ( j / 2. )
                if ( j <= 0 ):
                    break
            m = m + 1
#	Expand this bit pattern to separate components of the logical array INCLUD.
#
            j = poly[i-1]
            includ=zeros(m)

            for k in range(m, 0, -1):
                j2 = math.floor ( j / 2. )
                includ[k-1] =  (j != 2 * j2 )
                j = j2
##	Calculate the remaining elements of row I as explained
#	in Bratley and Fox, section 2.
#
            for j in range( m+1, maxcol+1 ):
                newv = v[i-1,j-m-1]
                l = 1
                for k in range(1, m+1):
                    l = 2 * l
                    if ( includ[k-1] ):
                        newv = bitwise_xor ( int(newv), int(l * v[i-1,j-k-1]) )
                v[i-1,j-1] = newv
##	Multiply columns of V by appropriate power of 2.
#
        l = 1
        for j in range( maxcol-1, 0, -1):
            l = 2 * l
            v[0:dim_num,j-1] = v[0:dim_num,j-1] * l
 
#	RECIPD is 1/(common denominator of the elements in V).   
        recipd = 1.0 / ( 2 * l )
        lastq=zeros(dim_num)
#
    seed = int(math.floor ( seed ))
    if ( seed < 0 ):
        seed = 0

    if ( seed == 0 ):
        l = 1
        lastq=zeros(dim_num)
    
    elif ( seed == seed_save + 1 ):
        l = i4_bit_lo0 ( seed )
    elif ( seed <= seed_save ):

        seed_save = 0
        l = 1
        lastq=zeros(dim_num)

        for seed_temp in range( int(seed_save), int(seed)):
            l = i4_bit_lo0 ( seed_temp )
            for i in range(1 , dim_num+1):
                lastq[i-1] = bitwise_xor ( int(lastq[i-1]), int(v[i-1,l-1]) )

        l = i4_bit_lo0(seed)

    elif ( seed_save + 1 < seed ):

        for seed_temp in range( int(seed_save + 1), int(seed) ):
            l = i4_bit_lo0 ( seed_temp )
            for i in range(1, dim_num+1):
                lastq[i-1] = bitwise_xor ( int(lastq[i-1]), int(v[i-1,l-1]) )

        l = i4_bit_lo0 ( seed )
#	Check that the user is not calling too many times!

    if ( maxcol < l ):
        print('I4_SOBOL - Fatal error!')
        print('	Too many calls!')
        print( '	MAXCOL = %d\n'%maxcol)
        print( '	L =			%d\n'%l)
        return
#
#	Calculate the new components of QUASI.
#
    quasi=zeros(dim_num)
    for i in range( 1, dim_num+1):
        quasi[i-1] = lastq[i-1] * recipd
        lastq[i-1] = bitwise_xor ( int(lastq[i-1]), int(v[i-1,l-1]) )

    seed_save = seed
    seed = seed + 1

    return [quasi, seed]




def i4_uniform ( a, b, seed ):
    #
    if ( seed == 0 ):
        print( 'I4_UNIFORM - Fatal error!') 
        print( '	Input SEED = 0!')
    
    seed = math.floor ( seed )
    a = round ( a )
    b = round ( b )
    
    seed = np.mod(seed, 2147483647)

    if ( seed < 0 ) :
        seed = seed + 2147483647

    k = math.floor ( seed / 127773 )

    seed = 16807 * ( seed - k * 127773 ) - k * 2836

    if ( seed < 0 ):
	       seed = seed + 2147483647

    r = seed * 4.656612875E-10
#
#	Scale R to lie between A-0.5 and B+0.5.
#
    r = ( 1.0 - r ) * ( min ( a, b ) - 0.5 ) + r * ( max ( a, b ) + 0.5 )
#
#	Use rounding to convert R to an integer between A and B.
#
    value = round ( r )

    value = max ( value, min ( a, b ) )
    value = min ( value, max ( a, b ) )

    c = value

    return [ int(c), int(seed) ]




def prime_ge ( n ):
    p = max ( math.ceil ( n ), 2 )
    while ( not isprime ( p ) ):
        p = p + 1

    return p


def isprime(n):
    if n!=int(n) or n<1:
	       return False
    p=2
    while p<n:
        if n%p==0:
            return False
        p+=1

    return True


#------------------------------------------------

def sobol_generate(k,N,skip,leap):
    """
    Skip and leap sobol sequence
    
    Reference:

    .. [1] Saltelli, A., Chan, K., Scott, E.M., "Sensitivity Analysis"
    """
    
    # Generate sobol sequence
    samples = i4_sobol_generate(k,N*(leap+1),skip).T;
    
    # Remove leap values
    samples = samples[0:samples.shape[0]:(leap+1),:]
    
    return samples


def scrambled_sobol_generate(k,N,skip,leap):
    """
    Scramble function as in Owen (1997)
    
    Reference:

    .. [1] Saltelli, A., Chan, K., Scott, E.M., "Sensitivity Analysis"
    """
    
    # Generate sobol sequence
    samples = sobol_generate(k,N,skip,leap);
    
    # Scramble the sequence
    for col in range(0,k):
        samples[:,col] = scramble(samples[:,col]);
    
    return samples


def scramble(X):
    """
    Scramble function as in Owen (1997)
    
    Reference:

    .. [1] Saltelli, A., Chan, K., Scott, E.M., "Sensitivity Analysis"
    """

    N = len(X) - (len(X) % 2)
    
    idx = X[0:N].argsort()
    iidx = idx.argsort()
    
    # Generate binomial values and switch position for the second half of the array
    bi = stats.binom(1,0.5).rvs(size=N/2).astype(bool)
    pos = stats.uniform.rvs(size=N/2).argsort()
    
    # Scramble the indexes
    tmp = idx[0:N/2][bi];
    idx[0:N/2][bi] = idx[N/2:N][pos[bi]];
    idx[N/2:N][pos[bi]] = tmp;
    
    # Apply the scrambling
    X[0:N] = X[0:N][idx[iidx]];
    
    # Apply scrambling to sub intervals
    if N > 2:
        X[0:N/2] = scramble(X[0:N/2])
        X[N/2:N] = scramble(X[N/2:N])
    
    return X