# HololensPointRegistration
 a unity plugin for 3D-3D point registration for the Hololens
 
 Isotropic case assumes that each correspondence is already known, there is minimal gaussian, zero-mean noise, and the noise is the same across each dimension.
 
 Anisotropic only differs in that the noise is different for all or some of the axes. It uses a weighting matrix, which is just an inverted covariance matrix for each axis - this means that it weights axes which it is more sure about heavier than ones which have a high variance (this is useful, for example, if a tracking system has a higher uncertainty along the depth axis, which is often the case).
 
 
code adapted from matlab code in Iterative Solution for Rigid-Body Point-Based Registration with Anisotropic Weighting by Balachadran et. al.

only tested for hololens 1

Anisotropic case currently is crashing for some reason. Isotropic appears to be fully functional.

Readme is a work in progress (to come: set up guide, making it look nicer, ect)
