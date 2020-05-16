from setuptools import find_packages, setup

setup(
    name='pandas-transformers',
    packages=find_packages("src"),
    package_dir={"": "src"}, 
    version='0.1.0',
    description='Pandas equivalents of (e.g.) the beloved scikit-learn transformers.',
    author='Nedim Bayrakdar',
    license='MIT',
)
