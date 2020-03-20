#!/bin/sh 
echo $SLURM_TMPDIR
echo flake8
flake8 || exit 1
echo black
black . || exit 1
echo pydocstyle
pydocstyle src || exit 1
echo mypy
mypy --ignore-missing-imports --package src --package tests || exit 1
