import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="optcom",
    version="0.1.0",
    author="Sacha Medaer",
    author_email="sacha.medaer@fau.de",
    python_requires=">=3.7.0",
    description="Optical system simulation software",
    long_description=long_description,
    long_description_content_type="text/markdown",
#    url="https://github.com/pypa/sampleproject",
    license="GNU",
    packages=setuptools.find_packages(),
    include_package_data=True,	# controls whether non-code files are copied when package is installed
    install_requires=["scipy", "numpy", "matplotlib", "nptyping", "pillow"],
    classifiers=[
        "Programming Language :: Python :: 3",
	"Programming Language :: Python :: 3.7"
        "License :: OSI Approved :: GNU General Public License",
        "Operating System :: OS Independent",
    ],
)
