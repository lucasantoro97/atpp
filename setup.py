from setuptools import setup, find_packages
from setuptools.command.install import install
import os

# Read the contents of your README file
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read the dependencies from requirements.txt
def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename, 'r') as f:
        return f.read().splitlines()

requirements = parse_requirements('requirements.txt')

# Define a custom install command
class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        print("\nThank you for installing ATPP!")
        print("To stay updated with the latest developments and news, please visit:")
        print("https://lucasantoro97.github.io/cv/\n")

setup(
    name='atpp',
    version='0.1.1',
    description='Active Thermography Test Post-Processing',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Luca Santoro',
    author_email='luca.santoro@polito.it',
    url='https://github.com/lucasantoro97/atpp',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    cmdclass={
        'install': CustomInstallCommand,
    },
)
