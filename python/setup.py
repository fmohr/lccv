from setuptools import setup

setup(
  name = 'lccv',
  packages = ['lccv'],
  version = '0.1.1',
  license='MIT',
  description = 'The official Learning Curve Database package',
  author = 'Felix Mohr',                   # Type in your name
  author_email = 'mail@felixmohr.de',      # Type in your E-Mail
  url = 'https://github.com/fmohr/lccv',   # Provide either the link to your github or to your website
  keywords = ['learning curves', 'sklearn', 'model selection', 'cross validation'],
  install_requires=[
          'numpy',
          'scikit-learn',
          'tqdm',
          'matplotlib',
          'func_timeout',
          'pandas'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9'
  ],
  package_data={'': []},
  include_package_data=False
)
