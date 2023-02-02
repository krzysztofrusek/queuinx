from setuptools import find_namespace_packages
from setuptools import setup


def _get_version():
  with open('queuinx/__init__.py') as fp:
    for line in fp:
      if line.startswith('__version__') and '=' in line:
        version = line[line.find('=')+1:].strip(' \'"\n')
        if version:
          return version
    raise ValueError('`__version__` not defined in `queuinx/__init__.py`')


setup(
    name='queuinx',
    version=_get_version(),
    url='https://github.com/krzysztofrusek/queuinx',
    license='Apache 2.0',
    author='Krzysztof Rusek',
    description=('Queuinx: A library for performance evaluation in Jax'),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author_email='krussek@gmail.com',
    keywords='jax graph neural networks python machine learning queuing theory buffer networking',
    packages=find_namespace_packages(exclude=['*_test.py']),
    python_requires='>=3.10',
    install_requires=[
        'jax>=0.3.25',
        'chex~=0.1.5'
    ],
    extras_require={'examples': ['jaxopt'],
                    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: System :: Networking'
    ],
)