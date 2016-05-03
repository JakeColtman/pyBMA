from os import path
from setuptools import setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
#with open(path.join(here, 'README.md')) as f:
#    long_description = f.read()

setup(
    name='pyBMA',
    packages=["pyBMA"],
    version='0.1.dev12',
    description='Bayesian Model Averaging in Python',
    long_description="Bayesian Model Averaging in Python",
    url='https://github.com/JakeColtman/pyBMA',
    author='Jake Coltman, Jacob Goodwin',
    author_email='jakecoltman@sky.com',
    install_requires=[
        'lifelines',
        'pandas',
        'numpy'
    ]
)
