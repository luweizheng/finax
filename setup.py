import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.0.1'
PACKAGE_NAME = 'jax-quant-finance'
AUTHOR = 'luweizheng'
AUTHOR_EMAIL = 'luweizheng36@hotmail.com'
URL = 'https://github.com/luweizheng/jax-quant-finance'

LICENSE = 'GPLv3'
DESCRIPTION = 'High Performance Quantative Finance Library on JAX'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = ['numpy', 'jax>=0.3.4', 'jaxlib>=0.3.2']
TESTS_REQUIRES = ['pytest']

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    author=AUTHOR,
    license=LICENSE,
    author_email=AUTHOR_EMAIL,
    url=URL,
    install_requires=INSTALL_REQUIRES,
    extras_require={
        'tests': TESTS_REQUIRES,
        'complete': INSTALL_REQUIRES + TESTS_REQUIRES,
    },
    packages=find_packages(),
    python_requires='>=3'
)