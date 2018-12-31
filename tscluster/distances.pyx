"""
# distutils: language=c++
"""
from libc.math cimport sqrt

def euclid_dist(t1,t2):
    return sqrt(sum((t1-t2)**2))

