from setuptools import setup, find_packages

setup(
  name = 'torch-cfd',
  packages=find_packages(include=['torch_cfd', 'torch_cfd.*']),
  version='{{VERSION_PLACEHOLDER}}',
  license='Apache-2.0',
  description = 'PyTorch CFD',
  long_description='PyTorch Computational Fluid Dynamics Library',
  long_description_content_type="text/markdown",
  author = 'Shuhao Cao',
  author_email = 'scao.math@gmail.com',
  url = 'https://github.com/scaomath/torch-cfd',
  keywords = ['pytorch', 'cfd', 'pde', 'spectral', 'fluid dynamics', 'deep learning', 'neural operator'],
  python_requires='>=3.10',
  install_requires=[
      'numpy>=1.24.0',
      'torch>=2.5.0',
  ],
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Science/Research',
      'Topic :: Scientific/Engineering :: Mathematics',
      'License :: OSI Approved :: Apache Software License',
      'Programming Language :: Python :: 3.10',
  ],
)
