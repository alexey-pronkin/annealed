# !/usr/bin/env python

import os
from distutils.core import setup

folder = os.path.dirname(os.path.realpath(__file__))
requirements_path = os.path.join(folder, 'requirements.txt')
install_requires = []
if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()

setup(name='avo',
      version='0.1',
      description='Bayesian methods final project',
      author='Mikhail Kurenkov, Aleksei Pronkin, Timur Chikichev',
      author_email='Mikhail.Kurenkov@skoltech.ru',
      package_dir={},
      packages=["avo", "avo.models", "avo.data_modules"],
      install_requires=install_requires
      )
