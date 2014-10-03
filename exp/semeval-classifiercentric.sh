#!/bin/bash

THREADS=10
LANGPAIRS="en-es en-de fr-en nl-en"
CORPUSNAME="ep7os12"
BINDIR="/vol/customopt/machine-translation/bin"
MOSESDIR="/vol/customopt/machine-translation/src/mosesdecoder/"
EXPDIR="/scratch/proycon/colibrita/semeval/"

cd $EXPDIR

for LANGPAIR in $LANGPAIRS; do
    L1=${LANGPAIR:0:2}
    L2=${LANGPAIR:3:2}
    echo $L1 $L2
    cd $LANGPAIR
    if [ $? -ne 0 ]; then
        echo "No such dir: $LANGPAIR" >&2
        break
    fi

    if [ ! -f $CORPUS-train.${L2}.lm ]; then
         ngram-count -text $CORPUS-train.${L2}.txt -order 3 -interpolate -kndiscount -unk -lm $CORPUS-train.${L2}.lm
    fi  

    mkdir train-$CORPUS
    cd train-$CORPUS
    ln -s ../$CORPUS-train.${L1}.txt train.$L1
    ln -s ../$CORPUS-train.${L2}.txt train.$L2


    $MOSESDIR/scripts/training/train-model.perl -external-bin-dir $BINDIR  -root-dir . --corpus train --f $L1 --e $L2 --first-step 1 --last-step 9 -reordering msd-bidirectional-fe --lm 0:3:$EXPDIR/$LANGPAIR/$CORPUS-train.en.lm -mgiza -mgiza-cpus $THREADS -core $THREADS -sort-buffer-size 10G -sort-batch-size 253 -sort-compress gzip -sort-parallel $THREADS


done
