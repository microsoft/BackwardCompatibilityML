#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
import setuptools


# this must be incremented every time we push an update to pypi (but not before)
VERSION = "1.4.0"

# supply contents of our README file as our package's long description
with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = []
with open("requirements.txt", "r") as fr:
    requirements = list(filter(
        lambda rq: rq != "",
        map(lambda r: r.strip(), fr.read().split("\n"))))

setuptools.setup(
    # this is the name people will use to "pip install" the package
    name="backwardcompatibilityml",

    version=VERSION,
    author="Besmira Nushi",
    author_email="benushi@microsoft.com",
    description="Project for open sourcing research efforts on Backward Compatibility in Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/BackwardCompatibilityML",

    # this will find our package "xtlib" by its having an "__init__.py" file
    packages=[
        "backwardcompatibilityml",
        "backwardcompatibilityml.loss",
        "backwardcompatibilityml.helpers",
        "backwardcompatibilityml.widgets",
        "backwardcompatibilityml.widgets.compatibility_analysis.resources",
         "backwardcompatibilityml.widgets.model_comparison.resources"

    ],  # setuptools.find_packages(),

    entry_points={
    },

    # normally, only *.py files are included - this forces our YAML file and
    # controller scripts to be included
    package_data={'': ['*.yaml', '*.sh', '*.bat', '*.txt', '*.rst', '*.crt', '*.json', '*.js', '*.html', '*.css']},
    include_package_data=True,

    # the packages that our package is dependent on
    install_requires=requirements,
    extras_require=dict(
        dev=[
            "sphinx==3.2.1",
            "recommonmark==0.6.0",
            "sphinx-rtd-theme==0.5.0"
        ], ),

    # used to identify the package to various searches
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
