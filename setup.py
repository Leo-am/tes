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
        'tes.analysis', 'tes.base', 'tes.data', 'tes.filesets', 'tes.maps', 
        'tes.mca', 'tes.protocol', 'tes.registers' 
    ],
    requires=['lmfit', 'matplotlib', 'numba', 'numpy', 'pyzmq',  'qutip', 'scipy' 
              ]
)
