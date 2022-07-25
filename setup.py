from setuptools import setup, find_packages
from pkg_resources import parse_requirements


with open("requirements.txt") as f:
    requirements = [str(req) for req in parse_requirements(f.read())]

setup(
    name="sentiment_classification",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages("sentiment_classification"),
    install_requires=requirements,
)
