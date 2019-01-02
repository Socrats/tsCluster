"""
# distutils: language=c++
"""
from libc.math cimport sqrt

cdef euclidean_dist(double[:] t1,t2):
    return sqrt(sum((t1-t2)**2))

cdef euclidean_dist(double[:] t1, double[:] t2)

cdef euclidean_dist(double[:] t1, double[:, :] t2)

cdef euclidean_dist(double[:, :] t1, double[:] t2)

