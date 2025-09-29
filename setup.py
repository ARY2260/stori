from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='stori',
    version='0.1.0',
    description='STORI (STOchastic-ataRI) benchmark and taxonomy for stochastic environments in reinforcement learning',
    license='MIT',
    long_description="""STORI (STOchastic-ataRI) is a benchmark and taxonomy for stochastic environments in reinforcement learning.""",
    author='Aryan Amit Barsainyan, Jing Yu Lim, Dianbo Liu',
    author_email='aryan.barsainyan@gmail.com, jy_lim@comp.nus.edu.sg, dianbo@nus.edu.sg',
    packages=find_packages(),
    install_requires=requirements,
)
