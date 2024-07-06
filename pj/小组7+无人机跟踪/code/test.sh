#!/bin/bash

export PYTHONPATH=$PWD

python tools/test.py --dataset UAVTrack112 --snapshot ./snapshot/checkpoint00_e66.pth


for ((i=54; i<=100; i++))
do
    python tools/test.py --dataset UAV123 --snapshot ./snapshot/checkpoint00_e${i}.pth
    python tools/test.py --dataset DTB70 --snapshot ./snapshot/checkpoint00_e${i}.pth
    python tools/test.py --dataset UAVTrack112_L --snapshot ./snapshot/checkpoint00_e${i}.pth
    python tools/test.py --dataset UAVTrack112 --snapshot ./snapshot/checkpoint00_e${i}.pth
done
