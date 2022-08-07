from setuptools import setup, find_packages
from os import path


setup(name = 'pyzpc',
    packages=find_packages(),
    version = '0.0.1',
    description = 'Python library that implements ZPC: Zonotopic Data-Driven Predictive Control.',
    url = 'https://github.com/rssalessio/PyZPC',
    author = 'Alessio Russo',
    author_email = 'alessior@kth.se',
    install_requires=['numpy', 'scipy', 'cvxpy'],
    license='MIT',
    zip_safe=False,
    python_requires='>=3.7',
)