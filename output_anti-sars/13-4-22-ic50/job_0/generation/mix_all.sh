#!/bin/bash

totalFile="combined.smi"
touch $totalFile

for i in epoch*_0.smi 
do
    cat $i >> $totalFile
done