#!/bin/bash

cwd=$(pwd)
pwd
cd data/datasets/raw
wget https://jacquard.liris.cnrs.fr/data/Download/Jacquard_Samples.zip
unzip Jacquard_Samples.zip
mv Samples jacquard_samples
rm Jacquard_Samples.zip
cd $cwd
