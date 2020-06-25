import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="optcom",
    version="0.3.2",
    author="Sacha Medaer",
    author_email="sacha.medaer@optcom.org",
    python_requires=">=3.7.0",
    description="Optical System Simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/optcom-org/optcom",
    download_url='https://github.com/optcom-org/optcom/archive/v0.3.2.tar.gz',
    license='Apache License 2.0',
    packages=setuptools.find_packages(exclude=("tests",)),
    include_package_data=True,	# controls whether non-code files are copied when package is installed
    install_requires=["scipy", "numpy", "matplotlib", "pillow", "pyfftw",
                      "typing_extensions"],
    classifiers=[
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
)
