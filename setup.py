from setuptools import setup, find_packages

setup(
  name = "tenop",
  version = "0.0.1",
  author = "Sahil Rajwar",
  license = "MIT",
  description = "Lightweight tensor computation library with CUDA backend",
  long_description=open("README.md").read(),
  long_description_content_type="text/markdown",
  packages=find_packages(),
  package_data={"tenop.engine": ["*.so", "*.pyi"]},
  classifiers=[
    "Programming Language :: Python :: 3",
    "Operating System :: POSIX :: Linux",
  ],
  python_requires=">=3.6",
)

