#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
from os import path
from os import path
import sys

sys.path.insert(0, "flatwrm2")
from version import __version__

# Load requirements
requirements = None
with open('requirements.txt') as file:
    requirements = file.read().splitlines()

# If Python3: Add "README.md" to setup.
# Useful for PyPI. Irrelevant for users using Python2.
try:
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except:
    long_description = ' '

# Command-line tools
entry_points = {'console_scripts': [
    'flatwrm2 = flatwrm2:flatwrm2_from_commandline'
]}

desc="FLATW'RM2 is a deep learning code that was developed to detect flares in light curves obtained by space observatories"

setup(name='flatwrm2',
      description=desc,
      long_description=long_description,
      version=__version__,
      maintainer='Krisztian Vida, Attila Bodi',
      maintainer_email='vidakris@konkoly.hu',
      license='GNU GPLv3.0',
      url='https://github.com/vidakris/flatwrm2',
      packages=['flatwrm2'],
      include_package_data=True,
      install_requires=requirements,
      entry_points=entry_points,
      package_data={'': ['README.md', 'LICENSE'],
                    'flatwrm2': ['LSTM_weights_keplerSC_only.h5',
                                'LSTM_weights_keplerSC-kfold-0.h5',
                                'LSTM_weights_keplerSC-kfold-1.h5',
                                'LSTM_weights_keplerSC-kfold-2.h5',
                                'LSTM_weights_keplerSC-kfold-3.h5',
                                'LSTM_weights_keplerSC-kfold-4.h5',
                                'LSTM_weights_keplerSC.h5']},
      classifiers=[ 'Topic :: Scientific/Engineering :: Astronomy']
)
