#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='xbrainmap',
    packages=find_packages(exclude=['test*']),
    version=open('VERSION').read().strip(),
    include_package_data=True,
    zip_safe=False,
    author='Doga Gursoy',
    author_email='dgursoy@aps.anl.gov',
    description='xbrainmap',
    keywords=['x-ray', 
              'brain',
              'tomography',
              'imaging'],
    url='http://xbrainmap.readthedocs.org',
    download_url='http://github.com/xbrainmap/xbrainmap.git',
    license='BSD-3',
    platforms='Any',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: Linux',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5']
)
