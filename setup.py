import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="keras_for_biology",
    version="0.1",
    author="Yosuke Toda",
    author_email="tyosuke@aquaseerser.com",
    description="Deep learning examples with biological themes using keras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)