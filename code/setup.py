from setuptools import setup, find_packages

# Read the dependencies from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='Uniforce', version='1.0', packages=find_packages(), install_requires=requirements)

# Navigate to the code folder and execute 'pip install -e .'
