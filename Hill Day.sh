#!/bin/bash
# Script to start image collection
# Increases the priority of the process since it has realtime components

cd ~/imagecapturegui

while [ true ] ; do
  nice -20 /usr/bin/python ./hill-day-demo.py
  sleep 5
done
