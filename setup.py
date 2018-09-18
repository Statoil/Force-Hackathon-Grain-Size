# -*- coding: utf-8 -*-

# from setuptools import setup
from distutils.core import setup
from Cython.Build import cythonize

setup(name='core_photo_force',
      version='0.0.1',
      description='Package for core photos in the FORCE hackathon',
      url='https://github.com/Statoil/Force-Hackathon-Grain-Size',
      author='Force Hackathon Core Photo Team',
      author_email='nathan.geology@gmail.com',
      license='BSD Open Source',
      packages=['core_photo_force'],
      install_requires=['lasio',
                        'scikit-image'
                        ],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],
      package_data={'core_photo_force': ['data/*.csv',
                               'packageData/*.pkl']})
