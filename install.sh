#!/bin/bash
pip uninstall tenop -y
pip install .
rm -rf build/ dist/ *.egg-info/
