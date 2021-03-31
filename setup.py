from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pysloth",
    version="0.0.2",
    packages=find_packages(exclude=["etc", "tests", "docs"]),
    description="Probabilistic Predictions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PySloth/pysloth",
    author="Valery Manokhin",
    author_email="valery.manokhin.2015@live.rhul.ac.uk",
    maintainer="leonarduschen",
    maintainer_email="leonardus.chen@gmail.com",
    license="MIT",
    scripts=[],
    install_requires=[
        "statsmodels>=0.12.2",
        "scikit-learn>=0.24.1",
        "numpy>=1.19.2",
    ],
    tests_require=["pytest"],
    setup_requires=['pytest-runner'],
    include_package_data=True,
    data_files=[("", [])],
)
