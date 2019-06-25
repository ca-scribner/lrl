import os
import sys

from setuptools import setup

from .lrl.version import VERSION

# 'setup.py publish' shortcut.
# Thanks to requests for the idea and code
if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist bdist_wheel')
    os.system('twine upload dist/*')
    sys.exit()


def get_readme():
    """
    Get readme for long description
    """
    with open('README.md') as fin:
        return fin.read()


setup(name='lrl',
      version=VERSION,
      description='lrl: Learn Reinforcement Learning',
      long_description=get_readme(),
      long_description_content_type='text/markdown',
      author='Andrew Scribner',
      license='BSD 3',
      url='https://github.com/ca-scribner/lrl',
      packages=['lrl'],
      # classifiers=[],  # Where do these come from?
      install_requires=[
          'gym>=0.12.1',
          'matplotlib>=3.0.3',
          'ipython>=7.5.0',
          'pandas>=0.24.2',
          'scikit-learn>=0.20.3'
      ],
      python_requires='>3.5',
      zip_save=False,
      )
