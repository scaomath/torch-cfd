from setuptools import setup, find_packages

setup(
  name = 'torch-cfd',
  packages=find_packages(include=['torch_cfd', 'torch_cfd.*']),
  version = '0.0.1',
  license='Apache-2.0',
  description = 'PyTorch CFD',
  long_description='PyTorch Computational Fluid Dynamics Library',
  long_description_content_type="text/markdown",
  author = 'Shuhao Cao',
  author_email = 'scao.math@gmail.com',
  url = 'https://github.com/scaomath/torch-cfd',
  keywords = ['pytorch', 'cfd', 'pde', 'spectral'],
  install_requires=[
      'numpy',
      'torch>=2.0.1',
  ],
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Science/Research',
      'Topic :: Scientific/Engineering :: Mathematics',
      'License :: OSI Approved :: Apache Software License',
      'Programming Language :: Python :: 3.8',
  ],
)
