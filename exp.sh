#!/bin/bash

#full experiment pipeline

#corpus should already be tokenised!
if [ $# -ne 6 ]; then
  echo "Syntax: exp.sh sourcecorpus targetcorpus setsize sourcelang targetlang expname" >&2
  exit 2
fi

SOURCECORPUS=$1
TARGETCORPUS=$2
SETSIZE=$3
SOURCELANG=$4
TARGETLANG=$5
EXPNAME=$6


SEED=12345

#only for cross-validation
CLONES=16
FOLDS=10

MOSESDIR=/vol/customopt/machine-translation/src/mosesdecoder
BINDIR=/vol/customopt/machine-translation/bin
SAMPLER=/vol/customopt/uvt-ru/src/pynlpl/tools/sampler.py #part of pynlpl
MATREXDIR=/vol/customopt/machine-translation/mtevalscripts

TRAINSET=$EXPNAME.train.xml
TESTSET=$EXPNAME.test.xml
TESTSETSIZE=5000


function forkconfig {
  mkdir $1
  cd $1
  find ../exp-l5r5k/ -name "*.train" | xargs -I{} cp -s {} .
  find ../exp-l5r5k/ -name "*.keywords" | xargs -I{} cp -s {} .
  find ../exp-l5r5k/ -name "*.conf" | xargs -I{} cp -s {} .
  cd ..
}


EXPDIR=$EXPNAME-$SOURCELANG-$TARGETLANG
mkdir $EXPDIR
cd $EXPDIR
LM=$EXPNAME-$TARGETLANG.srilm


echo "===== Splitting training and test set =====" >&2
$SAMPLER -d $SETSIZE -t $SETSIZE -S $SEED $SOURCECORPUS $TARGETCORPUS || exit 2

mv $SOURCECORPUS.dev $EXPNAME-$SOURCELANG.train.txt  #no bug, cheating using dev as train 
mv $TARGETCORPUS.dev $EXPNAME-$TARGETLANG.train.txt  #no bug 
mv $SOURCECORPUS.test $EXPNAME-$SOURCELANG.test.txt   
mv $TARGETCORPUS.test $EXPNAME-$TARGETLANG.test.txt   
mv $SOURCECORPUS.train $EXPNAME-$SOURCELANG.dev.txt   
mv $TARGETCORPUS.train $EXPNAME-$TARGETLANG.dev.txt   

echo "===== Building training set =====" >&2
colibrita-setgen --train -p 0.01 -D 0.8 -O 2 --mosesdir=$MOSESDIR --bindir=$BINDIR -S $SOURCELANG -T $TARGETLANG --seed=$SEED -s $EXPNAME-$SOURCELANG.train.txt -t $EXPNAME-$TARGETLANG.train.txt -o $EXPNAME > $TRAINSET || exit 2

echo "===== Building test set =====" >&2
colibrita-setgen --test -p 0.01 -D 0.8 -O 2 --mosesdir=$MOSESDIR --bindir=$BINDIR -S $SOURCELANG -T $TARGETLANG --seed=$SEED -s $EXPNAME-$SOURCELANG.test.txt  -t $EXPNAME-$TARGETLANG.test.txt -o $EXPNAME -n $TESTSETSIZE > $TESTSET || exit 2

echo "===== Building language model =====" >&2
ngram-count -text $EXPNAME-$TARGETLANG.train.txt -order 3 -interpolate -kndiscount -unk -lm $LM || exit 2


echo "===== Building baseline =====" >&2
colibrita-baseline -t $TESTSET -T train-$EXPNAME/model/phrase-table.gz -o baseline || exit 2

echo "===== Building LM-informed baseline =====" >&2
colibrita --test -f $TESTSET --lm $LM -T train-$EXPNAME/model/phrase-table.gz -o lmbaseline || exit 2

echo "===== Extracting training instances =====" >&2
colibrita --igen -f $TRAINSET -l 5 -r 5 -k -o exp-l5r5k || exit 2

echo "===== Forking for different configurations =====" >&2
forkconfig exp-l1r1
forkconfig exp-l2r2
forkconfig exp-l3r3

forkconfig exp-l1r1k
forkconfig exp-l2r2k
forkconfig exp-l3r3k

forkconfig exp-l1r0
forkconfig exp-l2r0
forkconfig exp-l3r0


forkconfig exp-l2r1

forkconfig exp-al5r5k


echo "===== Running configuration l1r1 =====" >&2
colibrita --train -f $TRAINSET -l 1 -r 1 -o exp-l1r1 --trainfortest $TESTSET || exit 2
colibrita --test -f $TESTSET -l 1 -r 1 -o exp-l1r1 || exit 2
colibrita-evaluate --matrexdir $MATREXDIR --ref $TESTSET --out exp-l1r1.output.xml || exit 2

echo "===== Running configuration l1r1k =====" >&2
colibrita --train -f $TRAINSET -l 1 -r 1 -k -o exp-l1r1k --trainfortest $TESTSET  || exit 2
colibrita --test -f $TESTSET -l 1 -r 1 -k -o exp-l1r1k || exit 2
colibrita-evaluate --matrexdir $MATREXDIR --ref $TESTSET --out exp-l1r1k.output.xml || exit 2



echo "===== Running configuration l2r2 =====" >&2
colibrita --train -f $TRAINSET -l 2 -r 2 -o exp-l2r2 --trainfortest $TESTSET  || exit 2
colibrita --test -f $TESTSET -l 2 -r 2 -o exp-l2r2 || exit 2
colibrita-evaluate --matrexdir $MATREXDIR --ref $TESTSET --out exp-l2r2.output.xml || exit 2

echo "===== Running configuration l2r2k =====" >&2
colibrita --train -f $TRAINSET -l 2 -r 2 -k -o exp-l2r2k --trainfortest $TESTSET  || exit 2
colibrita --test -f $TESTSET -l 2 -r 2 -k -o exp-l2r2k || exit 2
colibrita-evaluate --matrexdir $MATREXDIR --ref $TESTSET --out exp-l2r2k.output.xml || exit 2


echo "===== Running configuration l3r3 =====" >&2
colibrita --train -f $TRAINSET -l 3 -r 3 -o exp-l3r3 --trainfortest $TESTSET  || exit 2
colibrita --test -f $TESTSET -l 3 -r 3 -o exp-l3r3 || exit 2
colibrita-evaluate --matrexdir $MATREXDIR --ref $TESTSET --out exp-l3r3.output.xml || exit 2


echo "===== Running configuration l3r3k =====" >&2
colibrita --train -f $TRAINSET -l 3 -r 3 -k -o exp-l3r3k --trainfortest $TESTSET  || exit 2
colibrita --test -f $TESTSET -l 3 -r 3 -k -o exp-l3r3k || exit 2
colibrita-evaluate --matrexdir $MATREXDIR --ref $TESTSET --out exp-l3r3k.output.xml || exit 2

echo "===== Running configuration l1r0 =====" >&2
colibrita --train -f $TRAINSET -l 1 -r 0 -o exp-l1r0 --trainfortest $TESTSET  || exit 2
colibrita --test -f $TESTSET -l 1 -r 0 -o exp-l1r0 || exit 2
colibrita-evaluate --matrexdir $MATREXDIR --ref $TESTSET --out exp-l1r0.output.xml || exit 2

echo "===== Running configuration l2r0 =====" >&2
colibrita --train -f $TRAINSET -l 2 -r 0 -o exp-l2r0 --trainfortest $TESTSET  || exit 2
colibrita --test -f $TESTSET -l 2 -r 0 -o exp-l2r0 || exit 2
colibrita-evaluate --matrexdir $MATREXDIR --ref $TESTSET --out exp-l2r0.output.xml || exit 2

echo "===== Running configuration l3r0 =====" >&2
colibrita --train -f $TRAINSET -l 3 -r 0 -o exp-l3r0 --trainfortest $TESTSET  || exit 2
colibrita --test -f $TESTSET -l 3 -r 0 -o exp-l3r0 || exit 2
colibrita-evaluate --matrexdir $MATREXDIR --ref $TESTSET --out exp-l3r0.output.xml || exit 2

echo "===== Running configuration l2r1 =====" >&2
colibrita --train -f $TRAINSET -l 2 -r 1 -o exp-l2r1 --trainfortest $TESTSET  || exit 2
colibrita --test -f $TESTSET -l 2 -r 1 -o exp-l2r1 || exit 2
colibrita-evaluate --matrexdir $MATREXDIR --ref $TESTSET --out exp-l2r1.output.xml || exit 2

echo "===== Running configuration l1r1lm =====" >&2
ln -s exp-l1r1 exp-l1r1lm
colibrita --test -f $TESTSET -l 1 -r 1 --lm $LM -o exp-l1r1lm || exit 2
colibrita-evaluate --matrexdir $MATREXDIR --ref $TESTSET --out exp-l1r1lm.output.xml || exit 2

echo "===== Running configuration l1r1klm =====" >&2
ln -s exp-l1r1k exp-l1r1llm
colibrita --test -f $TESTSET -l 1 -r 1 --lm $LM -o exp-l1r1klm || exit 2
colibrita-evaluate --matrexdir $MATREXDIR --ref $TESTSET --out exp-l1r1klm.output.xml || exit 2

echo "===== Running configuration l2r2lm =====" >&2
ln -s exp-l2r2 exp-l2r2lm
colibrita --test -f $TESTSET -l 2 -r 2 --lm $LM -o exp-l2r2lm || exit 2
colibrita-evaluate --matrexdir $MATREXDIR --ref $TESTSET --out exp-l2r2lm.output.xml || exit 2

echo "===== Running auto configuration =====" >&2
colibrita --train -f $TRAINSET -l 5 -r 5 -k -a --Tclones $CLONES --folds $FOLDS --trainfortest $TESTSET -o exp-al5r5k || exit 2
colibrita --test -f $TESTSET -l 5 -r 5 -k -a -o exp-al5r5k || exit 2
colibrita-evaluate --matrexdir $MATREXDIR --ref $TESTSET --out exp-al5r5k.output.xml || exit 2

echo "===== Running auto configuration with LM =====" >&2
ln -s exp-al5r5k exp-al5r5klm
colibrita --test -f $TESTSET -l 5 -r 5 -k -a --lm $LM -o exp-al5r5klm || exit 2
colibrita-evaluate --matrexdir $MATREXDIR --ref $TESTSET --out exp-al5r5klm.output.xml || exit 2




