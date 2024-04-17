#!/bin/bash

for experiment in {0..2}; do

    python3 main.py -config configs/params.yaml -experiment $experiment

done
