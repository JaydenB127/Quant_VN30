import os
from setuptools import setup

if os.path.exists("qlib"):
    import numpy
    from setuptools import Extension
    NUMPY_INCLUDE = numpy.get_include()
    ext_modules = [
        Extension(
            "qlib.data._libs.rolling",
            ["qlib/data/_libs/rolling.pyx"],
            language="c++",
            include_dirs=[NUMPY_INCLUDE],
        ),
        Extension(
            "qlib.data._libs.expanding",
            ["qlib/data/_libs/expanding.pyx"],
            language="c++",
            include_dirs=[NUMPY_INCLUDE],
        ),
    ]
else:
    ext_modules = []

setup(ext_modules=ext_modules)

