#! /usr/bin/env python
# -*- coding: utf8 -*-

import os
from setuptools import setup, find_packages



def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "colibrita",
    version = "0.3",
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
            'colibrita-datastats = colibrita.datastats:main',
            'colibrita-moses = colibrita.moses:main',
            'colibrita-setdiff = colibrita.setdiff:main',
            'colibrita-setselect = colibrita.setselect:main',
            'colibrita-setmerge = colibrita.setmerge:main',
            'colibrita-setshuffle = colibrita.setshuffle:main',
            'colibrita-setview = colibrita.setview:main',
            'colibrita-settok = colibrita.settok:main',
            'colibrita-setdist = colibrita.setdist:main',
            'colibrita-manualsetbuild = colibrita.manualsetbuild:main',
            'colibrita = colibrita.colibrita:main',
        ]
    },
    #include_package_data=True,
    #package_data = {'foliatools': ['*.xsl'] },
    install_requires=['colibricore >= 0.4.992', 'colibrimt >= 0.1.4', 'pynlpl >= 0.6.5', 'lxml >= 2.2']
)
