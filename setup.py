"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

setup(
    name='sexism_custom_classifier',  
    version='0.1',  
    author='Elif Alkac', 
    #package_dir={'': 'src'},  
    packages=find_packages(),
    #packages=find_packages(where='src'), 
    #py_modules=['src'],
    python_requires='>=3.7.8, <4',
    install_requires=['pandas', 'scikit-learn', 'nltk', 'tweet-preprocessor', 'pycontractions'],
    #entry_points={'console_scripts': ['fastep=fastentrypoints:main']},
    scripts = ['scripts/loadpackages.sh'],
)