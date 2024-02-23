from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="clover",
    version="0.1",
    description="Conformal Locally Valid interval-Estimates for Regression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="#",
    author="Luben Miguel Cruz Cabezas",
    author_email="lucruz45.cab@gmail.com",
    packages=["clover"],
    license="MIT",
    keywords=["prediction intervals", "conformal prediction", "local calibration"],
    install_requires=["numpy", "scikit-learn", "torch"],
    zip_safe=False,
)
