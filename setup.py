from setuptools import setup, find_packages

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pysloth",
    version="0.0.3",
    packages=find_packages(exclude=["etc", "tests", "docs"]),

    description="Probabilistic Predictions",
    long_description=long_description,
    long_description_content_type="text/x-rst",

    url="https://github.com/PySloth/pysloth",
    author="PySloth",
    author_email="pysloth.python@gmail.com",

    license="Apache Software License",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",

    ],
    install_requires=[
        "statsmodels>=0.12.2",
        "scikit-learn>=0.24.1",
        "numpy>=1.19.2",
    ],

    python_requires=">=3.6",
    tests_require=["pytest"],
    setup_requires=['pytest-runner']
)
