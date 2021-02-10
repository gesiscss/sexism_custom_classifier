from setuptools import setup, find_packages

setup(
    name='sexism_custom_classifier',  
    version='0.1',  
    author='Elif Alkac', 
    packages=find_packages(),
    python_requires='>=3.7.8, <4',
    install_requires=['pandas', 'scikit-learn', 'nltk', 'ipywidgets', 'tensorflow', 'transformers', 'spacy', 'bs4', 'unidecode', 'emoji']
)