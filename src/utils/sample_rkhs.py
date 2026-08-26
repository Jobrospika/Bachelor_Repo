#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:00:54 2023

@author: christian
"""

import numpy as np
import math

# For now, numpy variant
class PreRKHSfunction():
    """A function from the pre RKHS of a kernel, represented as a weighted sum 
    of partially evaluated kernels    
    
    """
    def __init__(self, kernel, base_points, coeffs):
        """Constructor for the PreRKHSfunction class
        
        Args:
            kernel:         A kernel object that behaves like scikit-learn covariance functions
            base_points:    A numpy n x d array, where n is the number of base points and d the dimension of the input space
            coeffs:         An array-like vector of length n, containing the coefficients for the function representation
        """
        # For now, we expect a kernel that behaves like the kernels from sklearn.gaussian_process.kernels
        self._kernel = kernel
        
        # Need shape n_inputs x n_dim
        self._base_points = base_points
        
        # Need shape n_inputs x 1
        self._coeffs = np.atleast_1d(coeffs).reshape([-1,1])
        
        # We precompute the kernel matrix (from the base points) for RKHS norm computation
        self._K = kernel(base_points)
        
    def __call__(self, x):
        """Evalutes the pre RKHS function at the given input(s)
        
        Args:
            x:  Input on which the function should be evaluated. If more than one input is to be evaluated,
                the argument should be a numpy array of shape m x d, where m is the number of inputs to be evaluated
                and d is the input dimension. If a 1d array-like vector of length m is used, it is interpreted as m x 1.
                
        Returns:
            The pre RKHS function evaluated on the inputs. If m inputs are evaluated, a 1d numpy array of length m is
            returned. If only one input is evaluated, a float is returned.
        """
        # Input 1d or n_eval x n_dim
        x = np.atleast_1d(x)
        if len(x.shape) == 1:
            x = x.reshape([-1,1])
            
        # From now on, work with n_eval x n_dim (scikit-learn compatible)
        result = np.sum(self._kernel(x, self._base_points) @ self._coeffs, axis=1)
        
        # Make it more convenient: If only one input is used, return a float instead of a numpy array of size 1
        if result.size == 1:
            return result.item()
        else:
            return result
    
    @property
    def rkhs_norm(self):
        """RKHS norm of the pre RKHS function
            
        Since the function is from the pre RKHS and represented as a weighted sum of partially evaluated kernels,
        the RKHS norm can be directly evaluated as the square root of the quadratic form of the coefficients with
        the kernel matrix.
        """
        return np.sqrt(self._coeffs.reshape([1,-1]) @ self._K @ self._coeffs.reshape([-1,1])).item()
    
    def scale_rkhs_norm(self, rkhs_norm):
        """Rescale the coefficients of the pre RKHS function to match a specified RKHS norm
        
        Args:
            rkhs_norm:  The new RKHS norm of the pre RKHS function   
        """
        self._coeffs = rkhs_norm/self.rkhs_norm*self._coeffs

class PreRKHSfunctionGenerator():
    """Generator for pre RKHS functions
    """
    def __init__(self, 
                 kernel, 
                 base_point_generator, 
                 coeff_generator=None,
                 n_base_points=None,
                 n_base_points_range=None):
        """Constructor for the generator class, sets the basic configuration
        
        Args:
            kernel: A kernel object that behaves like scikit-learn covariance functions
            base_point_generator: Callable with an optional integer argument n_base_points. When called without an argument
                it is supposed to return a n x d numpy array, where n is the number of base points (determined by the base_point_generator)
                and d the input dimension (has to be compatible with the kernel). If it is called with the argument n_base_points, it is
                supposed to return a n_base_points x d numpy array containing n_base_points base points of dimension d.
            coeff_generator: Optional. If set, it has to be a callable with one integer argument n_base_points, which returns a
                numpy array of length n_base_points and type float, containing the coefficients for the pre RKHS functions.
                Default is i.i.d. sampling from a standard normal distribution.
            n_base_points: Optional positive integer, fixes the number of coefficients in the weighted sum representing the function
            n_base_points_range: Optional, pair of two positive integers, restricting the minimum number (first element) 
                and maximum number (seoncd element) of coefficients (both inclusive).
        """
        self._kernel = kernel
        self._base_point_generator = base_point_generator
        
        # The default coefficient generator is i.i.d. sampling with a standard normal
        if coeff_generator is None:
            self._coeff_generator = lambda n_base_points: np.random.normal(size=n_base_points)
            
        # Deal with restrictions on number of coefficients
        if n_base_points_range is not None:
            self._n_base_points_range = n_base_points_range
            
        if n_base_points is not None:
            self._n_base_points_range = (n_base_points, n_base_points)

    
    def __call__(self, 
                 rkhs_norm=None, 
                 base_points=None, 
                 n_base_points=None):
        """Generates a pre RKHS function
        
        Args:
            rkhs_norm: Optional positive float, will force the generated pre RKHS function to have this RKHS norm
            base_points: Optional n x d numpy array containing n base points of dimension d that will be used as the base points
                of the generated RKHS function. The next argument will be ignored if this is set.
            n_point_points: Optional positive integer, fixes the number of base points generated.
        """
        # Get base points (where the kernel will be partially evaluated)            
        if base_points is None:
            # If no base points are given, we generate them using the base_point_generator
            # The default number of base points can be overwritten
            if n_base_points is None:
                # If not, use the default behaviour
                n_base_points = self._get_n_base_points()
                base_points = self._base_point_generator(n_base_points)
            else:
                base_points = self._base_point_generator(n_base_points)
        
        n_base_points = base_points.shape[0]
        
        # Get the coefficients
        coeffs = self._coeff_generator(n_base_points)
        
        # Build the pre RKHS function
        f_pre = PreRKHSfunction(self._kernel, base_points, coeffs)
        
        # At this point, the RKHS norm of the pre RKHS function is random. If a specific RKHS norm requested, we achieve
        # this by rescaling the coefficients (though this is done in the PreRKHSfunction class)
        if rkhs_norm is not None:
            f_pre.scale_rkhs_norm(rkhs_norm)
            
        return f_pre

    def _get_n_base_points(self):
        """Generates a number of base points (or None if no restrictions are set)
        
        Returns:
            Number of base points (positive integer) uniformly from the range n_base_points_range, 
            or None if no range has been specified
        """
        try:
            return np.random.choice(np.arange(self._n_base_points_range[0], self._n_base_points_range+1, dtype=np.int), size=1)
        except:
            return None


# ONB functions from Steinwart et al
def se_bfunc_1d(x, n, gamma=1):
    return np.sqrt(np.power(2,n)/(np.power(gamma, 2*n)*math.factorial(n))) * np.power(x, n) * np.exp(-x**2/gamma**2)