#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='tes',
    version='0.1.0',
    description="Package for communicating with the hardware processor "
                "connected to Transition Edge Sensors",
    author='Geoff Gillett',
    url='http://www.quantum.info',
    author_email='geoff.gillett@gmail.com',
    # package_dir={'tes': ''},
    py_modules=[
        'tes.base', 'tes.maps', 'tes.mca', 'tes.registers', 'tes.analysis'
    ],
    requires=['matplotlib', 'numpy', 'scipy', 'numba', 'qutip', 'lmfit',
              'pyzmq']
)
