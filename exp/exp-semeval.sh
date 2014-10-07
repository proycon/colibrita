#!/bin/bash

THREADS=10
LANGPAIRS="en-es en-de fr-en nl-en" #run this script with any of these values as parameter
CORPUSNAME="ep7os12"
BINDIR="/vol/customopt/machine-translation/bin"
MOSESDIR="/vol/customopt/machine-translation/src/mosesdecoder/"
EXPDIR="/scratch/proycon/colibrita/semeval3/"
TESTDIR="/home/proycon/semeval2014task5-pub/"
TRAINCONFIGURATIONS="l1r1 l2r1 l2r2 l3r3"
EVALCMD="semeval2014task5-evaluate --mtevaldir /vol/customopt/machine-translation/mtevalscripts"
cd $EXPDIR

LANGPAIR=$1
#for LANGPAIR in $LANGPAIRS; do

L1=${LANGPAIR:0:2}
L2=${LANGPAIR:3:2}
echo $L1 $L2
cd $LANGPAIR
if [ $? -ne 0 ]; then
    echo "No such dir: $LANGPAIR" >&2
    exit 2
fi

if [ ! -f $CORPUS-train.${L2}.lm ]; then
        ngram-count -text $CORPUS-train.${L2}.txt -order 3 -interpolate -kndiscount -unk -lm $CORPUS-train.${L2}.lm
fi  

mkdir train-$CORPUS
cd train-$CORPUS
ln -s ../$CORPUS-train.${L1}.txt train.$L1
ln -s ../$CORPUS-train.${L2}.txt train.$L2


if [ ! -f model/moses.ini ]; then
    $MOSESDIR/scripts/training/train-model.perl -external-bin-dir $BINDIR  -root-dir . --corpus train --f $L1 --e $L2 --first-step 1 --last-step 9 -reordering msd-bidirectional-fe --lm 0:3:$EXPDIR/$LANGPAIR/$CORPUS-train.en.lm -mgiza -mgiza-cpus $THREADS -core $THREADS -sort-buffer-size 10G -sort-batch-size 253 -sort-compress gzip -sort-parallel $THREADS

fi

if [ ! -f model/moses.ini ]; then
    echo "Moses failed!" >&2
    exit 2
fi

if [ ! -d mert-work ]; then
    $MOSESDIR/scripts/training/mert-moses.pl --mertdir=/vol/customopt/machine-translation/src/mosesdecoder/mert/ --decoder-flags="-threads $THREADS" dev.${L1}.txt dev.${L2}.txt `which moses` model/moses.ini --predictable-seeds --threads=$THREADS
fi

if [ ! -f mert-work/moses.ini ]; then
    echo "MERT failed!" >&2
    exit 2
fi

cd ..



for CONF in $TRAINCONFIGURATIONS; do
    L=${CONF:1:1}
    R=${CONF:3:1}
    if [ ! -d $CORPUS-$CONF ]; then
        colibrita --trainfortest $TESTDIR/corpus/$LANGPAIR.gold.tokenised.xml  --source $CORPUS-train.${L1}.txt --target $CORPUS-train.${L2}.txt -M train-$CORPUS/model/phrase-table.gz -l $L -r $R -o $CORPUS-$CONF -p 0.0001 
    fi

    if [ ! -d "$CORPUS-baseline" ]; then 
        ln -s $CORPUS-$CONF $CORPUS-baseline
        colibrita --baseline $TESTDIR/corpus/${LANGPAIR}.gold.tokenised.xml -o $CORPUS-baseline
        if [ "$?" -ne 0 ];
            echo "Failure in colibrita! ($CORPUS-baseline)" >&2
        else
            $EVALCMD --ref $TESTDIR/corpus/${LANGPAIR}.gold.tokenised.xml --out $CORPUS-baseline
        fi
    fi
    if [ ! -d $CORPUS-lmbaseline ]; then 
        ln -s $CORPUS-$CONF $CORPUS-baseline
        colibrita --baseline $TESTDIR/corpus/${LANGPAIR}.gold.tokenised.xml -o $CORPUS-lmbaseline --lm $CORPUS-train.${L2}.lm
        if [ "$?" -ne 0 ];
            echo "Failure in colibrita! ($CORPUS-lmbaseline)" >&2
        else
            $EVALCMD --ref $TESTDIR/corpus/${LANGPAIR}.gold.tokenised.xml --out $CORPUS-lmbaseline
        fi
    fi
    if [ ! -d "${CORPUS}-mosesbaseline" ]; then 
        ln -s $CORPUS-$CONF $CORPUS-mosesbaseline
        colibrita --test $TESTDIR/corpus/${LANGPAIR}.gold.tokenised.xml -Z -o $CORPUS-mosesbaseline --moseslm $CORPUS-train.${L2}.lm --mosesweights train-$CORPUS/mert-work/moses.ini
        if [ "$?" -ne 0 ];
            echo "Failure in colibrita! ($CORPUS-mosesbaseline)" >&2
        else
            $EVALCMD --ref $TESTDIR/corpus/${LANGPAIR}.gold.tokenised.xml --out $CORPUS-mosesbaseline
        fi
    fi

    colibrita --test $TESTDIR/corpus/${LANGPAIR}.gold.tokenised.xml -l $L -r $R -o $CORPUS-$CONF 
    if [ "$?" -ne 0 ];
        echo "Failure in colibrita! ($CORPUS-$CONF)" >&2
    else
        $EVALCMD --ref $TESTDIR/corpus/${LANGPAIR}.gold.tokenised.xml --out $CORPUS-$CONF
    fi
    
    if [ ! -d $CORPUS-${CONF}-lm ]; then
        ln -s $CORPUS-$CONF $CORPUS-${CONF}-lm
        colibrita --test $TESTDIR/corpus/${LANGPAIR}.gold.tokenised.xml -l $L -r $R -o $CORPUS-${CONF}-lm --lm $CORPUS-train.${L2}.lm 
        if [ "$?" -ne 0 ];
            echo "Failure in colibrita! ($CORPUS-$CONF-lm)" >&2
        else
            $EVALCMD --ref $TESTDIR/corpus/${LANGPAIR}.gold.tokenised.xml --out $CORPUS-$CONF-lm
        fi
    fi

    if [ ! -d $CORPUS-${CONF}-moses ]; then
        ln -s $CORPUS-$CONF $CORPUS-${CONF}moses
        colibrita --test $TESTDIR/corpus/${LANGPAIR}.gold.tokenised.xml -l $L -r $R -Z -o $CORPUS-${CONF}-moses --lm $CORPUS-train.${L2}.lm  --mosesweights train-$CORPUS/mert-work/moses.ini
        if [ "$?" -ne 0 ];
            echo "Failure in colibrita! ($CORPUS-$CONF-moses)" >&2
        else
            $EVALCMD --ref $TESTDIR/corpus/${LANGPAIR}.gold.tokenised.xml --out $CORPUS-$CONF-moses
        fi
    fi


    if [ ! -d $CORPUS-${CONF}-mosesweighted ]; then
        ln -s $CORPUS-$CONF $CORPUS-${CONF}-mosesweighted
        colibrita --test $TESTDIR/corpus/${LANGPAIR}.gold.tokenised.xml -l $L -r $R -X -o $CORPUS-${CONF}-mosesweighted --lm $CORPUS-train.${L2}.lm --mosesweights train-$CORPUS/mert-work/moses.ini 
        if [ "$?" -ne 0 ];
            echo "Failure in colibrita! ($CORPUS-$CONF-mosesweighted)" >&2
        else
            $EVALCMD --ref $TESTDIR/corpus/${LANGPAIR}.gold.tokenised.xml --out $CORPUS-$CONF-mosesweighted
        fi
    fi

    if [ ! -d $CORPUS-${CONF}-mosescut0.7 ]; then
        ln -s $CORPUS-$CONF $CORPUS-${CONF}-mosescut0.7
        colibrita --test $TESTDIR/corpus/${LANGPAIR}.gold.tokenised.xml -l $L -r $R -A 0.7 -o $CORPUS-${CONF}-mosescut0.7 --lm $CORPUS-train.${L2}.lm --mosesweights train-$CORPUS/mert-work/moses.ini 
        if [ "$?" -ne 0 ];
            echo "Failure in colibrita! ($CORPUS-$CONF-mosescut0.7)" >&2
        else
            $EVALCMD --ref $TESTDIR/corpus/${LANGPAIR}.gold.tokenised.xml --out $CORPUS-$CONF-mosescut0.7
        fi
    fi

    if [ ! -d $CORPUS-${CONF}-mosescut0.8 ]; then
        ln -s $CORPUS-$CONF $CORPUS-${CONF}-mosescut0.8
        colibrita --test $TESTDIR/corpus/${LANGPAIR}.gold.tokenised.xml -l $L -r $R -A 0.8 -o $CORPUS-${CONF}-mosescut0.8 --lm $CORPUS-train.${L2}.lm --mosesweights train-$CORPUS/mert-work/moses.ini 
        if [ "$?" -ne 0 ];
            echo "Failure in colibrita! ($CORPUS-$CONF-mosescut0.8)" >&2
        else
            $EVALCMD --ref $TESTDIR/corpus/${LANGPAIR}.gold.tokenised.xml --out $CORPUS-$CONF-mosescut0.8
        fi
    fi

    if [ ! -d $CORPUS-${CONF}-mosescut0.9 ]; then
        ln -s $CORPUS-$CONF $CORPUS-${CONF}-mosescut0.9
        colibrita --test $TESTDIR/corpus/${LANGPAIR}.gold.tokenised.xml -l $L -r $R -A 0.9 -o $CORPUS-${CONF}-mosescut0.9 --lm $CORPUS-train.${L2}.lm --mosesweights train-$CORPUS/mert-work/moses.ini 
        if [ "$?" -ne 0 ];
            echo "Failure in colibrita! ($CORPUS-$CONF-mosescut0.9)" >&2
        else
            $EVALCMD --ref $TESTDIR/corpus/${LANGPAIR}.gold.tokenised.xml --out $CORPUS-$CONF-mosescut0.9
        fi
    fi
done


#done
