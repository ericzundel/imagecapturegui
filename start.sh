#!/bin/bash
# Script to start image collection
# Increases the priority of the process since it has realtime components

python ./imagecapture-pi.py &
sudo renice -20 $!
