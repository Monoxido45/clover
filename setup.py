from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(name= "lcv",
version='0.1',
description = 'Local Calibration Validation of conformal prediction methods',
long_description=long_description,
long_description_content_type="text/markdown",
url='#',
author='Luben Miguel Cruz Cabezas',
author_email='lucruz45.cab@gmail.com',
packages = ["lcv"],
license='MIT',
keywords = ["conformal prediction", "hypothesis testing", "local calibration"],
install_requires = ["numpy", "scikit-learn", "torch"],
zip_safe=False
)