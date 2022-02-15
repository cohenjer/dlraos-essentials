#!/usr/bin/env python

import setuptools
import os

# Installing packages in requirements
thisfolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thisfolder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()


setuptools.setup(
      name='dlraos',
      version='0.0',
      description='Code for dictionary-based low-rank approximations (DLRA) with one-sparse coefficients. This code is an updated version from the 2018 publication `Dictionary-based CPD`.',
      author='Jeremy E. Cohen',
      author_email='jeremy.cohen@cnrs.fr',
      license='MIT',
      install_requires=install_requires
     )