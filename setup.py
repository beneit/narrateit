from setuptools import setup, find_namespace_packages

# Read requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="tpl",
    version="0.1",
    packages=find_namespace_packages(where="tpl"),
    package_dir={"": "tpl"},
    install_requires=required,
)