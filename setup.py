from setuptools import setup, find_packages
import os

# Read the contents of your README file
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read the dependencies from requirements.txt
def parse_requirements(filename):
    """ Load requirements from a pip requirements file """
    with open(filename, 'r') as f:
        return f.read().splitlines()

# This will load the requirements from requirements.txt
requirements = parse_requirements('requirements.txt')

setup(
    name='atpp',  # Package name
    version='0.1.0',
    description='Active Thermography Test Post-Processing',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Luca Santoro',
    author_email='luca.santoro@polito.it',
    url='https://github.com/lucasantoro97/atpp',
    packages=find_packages(),  # This will find 'script/' and include it
    include_package_data=True,
    install_requires=requirements,  # Read from requirements.txt
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
