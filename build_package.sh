#!/bin/bash

rm -r dist/*
python -m build
twine check dist/*
twine upload dist/*
