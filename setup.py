from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='tsCluster',
    version='0.0.1',
    packages=['crfs'],
    ext_modules=cythonize("crfs/crfsUnsupervised.pyx"),
    include_dirs=[np.get_include()],
    url='',
    license='GNU General Public License v3.0',
    author='Elias Fernandez',
    author_email='eliferna@vub.be',
    description='Clustering methods for time-series', install_requires=['Cython']
)
