#!/bin/bash

PATHS=". ./ui ./test"
for path in $PATHS
do
    for i in "$path/*.py"; do autopep8 --in-place $i; done
done    
	    

