#!/usr/bin/env python

from setuptools import setup, find_packages


with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="navigation",
    description="Udacity Banana collection project",
    author="Samuel Blais-Dowdy",
    author_email="samuel.blaisdowdy@protonmail.com",
    packages=find_packages(),
    install_requires=required,
)
