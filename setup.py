from setuptools import setup, find_packages

setup(
	name='skl_plus',
	version='0.1',
	description='Custom ML models and enhancements over scikit-learn',
	author='ANSH',
	packages=find_packages(),
	install_requires=['scikit-learn==1.6.1', 'numpy==2.2.2']
)