#!/bin/bash

seq_lens=(5 10)
archs=(0 1)

for seq_len in "${seq_lens[@]}"; do
    for arch in "${archs[@]}"; do
        python3 main.py -config configs/params.yaml -seq_len "$seq_len" -arch "$arch"
    done
done
