#!/usr/bin/env bash

SCDIR="../PROPER-FM-shared-component"
ENVPYTHON="/opt/anaconda3/envs/proper-fm/bin/python"

mkdir $SCDIR
mkdir $SCDIR/models
mkdir $SCDIR/tests

# bhild things
cp .gitignore pyproject.toml requirements.txt $SCDIR

# models and tests
cp ./models/*.py $SCDIR/models
cp ./tests/*.py $SCDIR/tests

# run tests in new directory
cd $SCDIR
$ENVPYTHON -m pytest --cov=models
cd -