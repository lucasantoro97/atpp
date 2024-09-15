from setuptools import setup, find_packages

setup(
    name='atpp',  # Package name
    version='0.1.0',
    description='Active Thermography Test Post-Processing',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='youremail@example.com',
    url='https://github.com/yourusername/atpp',
    packages=find_packages(),  # This will find 'script/' and include it
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
