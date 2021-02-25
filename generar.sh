#!/bin/bash
for filename in *.pdf; do
 	echo $filename
    python3 ocrtest.py $filename
done

