#!/bin/bash

for neg_sam in 1 2 3
do
	python execute.py -n_neg_sam ${neg_sam}
done
