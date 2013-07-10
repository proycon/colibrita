#! /usr/bin/env python
# -*- coding: utf8 -*-

import os
from setuptools import setup, find_packages



def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "colibrita",
    version = "0.1.1",
    author = "Maarten van Gompel",
    author_email = "proycon@anaproy.nl",
    description = (""),
    license = "GPL",
    keywords = "nlp computational_linguistics linguistics",
    url = "https://github.com/proycon/colibrita",
    packages=['colibrita'],
    #long_description=read('README'),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Text Processing :: Linguistic",
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    entry_points = {
        'console_scripts': [
            'colibrita-setgen = colibrita.setgen:main',
            'colibrita-evaluate = colibrita.evaluation:main',
            'colibrita-baseline = colibrita.baseline:main',
            'colibrita = colibrita:main',
        ]
    },
    #include_package_data=True,
    #package_data = {'foliatools': ['*.xsl'] },
    install_requires=['pynlpl >= 0.6.5', 'lxml >= 2.2']
)
