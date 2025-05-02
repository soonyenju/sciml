#!/usr/bin/env python
#-*- coding:utf-8 -*-

#############################################
# File Name: setup.py
# Author: Songyan Zhu
# Mail: zhusy93@gmail.com
# Created Time:  2021-06-09 08:22
#############################################


from setuptools import setup, find_packages

setup(
	name = "sciml",
	version = "0.0.12",
	keywords = ("Geospatial scientific ML"),
	description = "Machine/deep learning models and toolboxes for geosciences.",
	long_description = "coming soon",
	license = "MIT Licence",

	url="https://github.com/soonyenju/sciml",
	author = "Songyan Zhu",
	author_email = "zhusy93@gmail.com",

	packages = find_packages(),
	include_package_data = True,
	platforms = "any",
	install_requires=[

	]
)