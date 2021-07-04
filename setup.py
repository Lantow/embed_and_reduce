#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
from setuptools import setup

setup(
    name = 'embed_and_reduce',
    version='1.0',
    license='Creative Commons Legal Code',
    author='Morten Lantow',
    author_email='MrLantow@Gmail.com',
    description='Embeds text and dimentionality reduces to 100 dim',
    packages=['embed_and_reduce'],
    platforms='any',
    install_requires=[
        'flask',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'License :: Creative Commons Legal Code',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Internet :: WWW/HTTP :: Dynamic Content',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
)