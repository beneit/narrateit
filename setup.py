from setuptools import setup, find_namespace_packages

# Read requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="narrateit",
    version="0.1",
    packages=find_namespace_packages(where="narrateit"),
    package_dir={"": "narrateit"},
    install_requires=required,
)
