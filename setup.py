from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        r'admm_utils',
        [r'admm_utils.pyx'],
        include_dirs=[numpy.get_include()],
    ),
]

setup(
    name='admm_tools',
    ext_modules=cythonize(ext_modules),
)
