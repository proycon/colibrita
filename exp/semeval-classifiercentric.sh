#!/bin/bash

THREADS=10
LANGPAIRS="en-es en-de fr-en nl-en"
CORPUSNAME="ep7os12"
EXPDIR="/scratch/proycon/colibrita/semeval/"

cd $EXPDIR

for LANGPAIR in $LANGPAIRS; do
    L1=${LANGPAIR:0:2}
    L2=${LANGPAIR:3:2}
    mkdir $LANGPAIR
    echo $L1 $L2

done
