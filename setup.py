from setuptools import setup, find_packages

setup(
    name="pysloth",
    version="0.0.1",
    packages=find_packages(exclude=["etc", "tests", "docs"])
)
