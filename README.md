Colibrita: Translation Assistance System
============================================

Colibrita is a proof-of-concept translation assistance system that can
translate L1 fragments in an L2 context. 

The system is designed prior to a new task (presented at SemEval 2014) concerning the translation of L1
fragments, i.e words or phrases, in an L2 context. This type of translation can
be applied in writing assistance systems for language learners in which users
write in their target language, but are allowed to occasionally back off to
their native L1 when they are uncertain of the proper word or expression in L2.
These L1 fragments are subsequently translated, along with the L2 context, into
L2 fragments.

Colibrita was developed to test whether L2 context information aids in translation of L1 fragments. The results are accepted for publication in ACL 2014, in the paper:

    Maarten van Gompel, Antal van den Bosch. Translation Assistance by Translation of L1 Fragments in an L2 Context. Proceedings of ACL 2014 Conference (to appear still)


Installation
===============

Colibrita is written in Python 3. It is a complex system involving quite a
number of dependencies.

First make sure you have a modern linux distribution with the necessary
prerequisites: python3, python3-dev, python3-setuptools, python3-lxml, cython3, gcc,
g++,  autoconf,  automake, autoconf-archive, libtool , libboost-dev, libboost-python

If you intend to build your own training models, then you will also require the
following two dependencies:
    * **Moses** - https://github.com/moses-smt/mosesdecoder
    * **GIZA++** - http://code.google.com/p/giza-pp/ 

Other unix systems including FreeBSD and Mac OS X will most likely work 
too, but especially for that latter considerable extra effort may be required
in installing things. The instructions here have been tailored for
Debian/Ubuntu-based Linux distributions.

In addition to the above dependencies, Colibrita depends on pynlpl, colibri-core, Timbl and python-timbl.

Install PyNLPl from the Python Package Index (or alternatively from
https://github.com/proycon/pynlpl):

    $ sudo easy_install3 pynlpl

Download colibri-core from https://github.com/proycon/colibri-core and install as follows:

    $ bash bootstrap
    $ ./configure 
    $ make
    $ sudo make install
    $ sudo python3 ./setup.py install

Install Timbl, it may be in your package manager if you use Debian/Ubuntu:

    $ sudo apt-get install timbl

Otherwise obtain it from http://ilk.uvt.nl/timbl and compile manually:

    $ ./configure
    $ make
    $ make install

Install Python-Timbl from the Python Package Index (or alternatively from
https://github.com/proycon/python-timbl):

    $ sudo easy_install3 python-timbl

Then install colibrita from https://github.com/proycon/colibrita:

     $ sudo python3 ./setup.py install

**Note:** If you want to reproduce the results of our ACL paper, then make sure to do
``git checkout v0.2.1`` in the Colibrita repository prior to installation. Colibrita may have advanced
since then.

Last, if you want to evaluate according to well-known MT metrics such as BLEU,
METEOR, NIST, TER, WER, and PER; you should download and unpack
http://lst.science.ru.nl/~proycon/mtevalscripts.tar.gz


Usage
===========

The following tools are available:

 * ``colibrita`` - This is the main system, it is used for training and
   testing.
  
 * ``colibrita-evaluate`` - Tool for evaluation of system output. Point
   --mtevaldir to the directory where you unpacked mtevalscripts.tar.gz if you
   want common MT metrics in your report.

 * ``colibrita-setgen`` - Tool for generating training & test sets from
   parallel corpus data, GIZA++ Word Alignments and a Moses Phrasetable
    
 * ``colibrita-datastats`` - Reports some statistics on a dataset (train or test, XML)

 * ``colibrita-manualsetbuild`` - Small interactive console-based script for creating
   datasets manually

Set generation
--------------

Building a model starts with generating a training set from a parallel corpus.
Ensure you have two plain-text files, one in the source language, one in the
target language, with one sentence per line where the line numbers across the
two files are indicative of sentences that are translations of eachother. In
this documentation we will use two files from our ACL 2014 experiments,
obtainable from http://lst.science.ru.nl/~proycon/colibrita-acl2014-data.zip :

    * europarl200k-train.nl.txt
    * europarl200k-train.en.txt

Given this input data, you can use Colibrita's *setgen* tool: 

     $ colibrita-setgen --train --mosesdir=/path/to/mosesdecoder -S nl -T en -s europarl200k-train.nl.txt -t europarl200k-train.en.txt --bindir=/usr/local/bin -o europarl200k

This tool will invoke Moses (which will in turn invoke GIZA++) and the
Colibri-Core patternmodeller. It builds word alignments, a phrase-translation
table and pattern models, and eventually produces an XML file. This process may
take a very long time and demands conseridable memory. The output prefix
 ``-o`` will be used in many of the output files. The parameters
``--joinedprobabilitythreshold`` and ``--divergencefrombestthreshold`` can be
used to prevent weaker alignments and alternatives from making it into the set,
and correspond to the parameters λ1 and λ2 in our ACL 2014 paper.

A test set can be generated in the same fashion:

     $ colibrita-setgen --test --mosesdir=/path/to/mosesdecoder -S nl -T en -s europarl200k-test.nl.txt -t europarl200k-test.en.txt --bindir=/usr/local/bin -o europarl200k


Training
--------------

The next step is feature extraction and classifier training:

    $ colibrita --train -f europarl200k.train.xml -l 1 -r 1 -o exp-l1r1 --Tclones 4 --trainfortest europarl200k-test.xml


The output will consist of a whole bunch of classifiers (ibase files) in the
directory specified with ``-o``.

Some notes about this example:

 * ``-f`` specifies the training set, generated by ``colibrita-setgen`` in the
   previous step.
 * ``-l 1`` sets a left context size of one
 * ``-r 1`` sets a right context size of one
 * ``-o`` specified a new output prefix, used in generated files and a
   directory will be generated with this name containing all classifiers and
   intermediate files
 * ``--Tclones 4`` runs Timbl on four cores
 * ``--trainfortest`` generates only those classifiers that will be used in
   testing, saving time and resources. But this implies the model will have to
   be retrained if other test data is offered, and can ever be used in a
   live setting.

Testing
- - - - - - 

Testing follows a very similar syntax:

    $ colibrita --test -f europarl200k.test.xml -l 1 -r 1 -o exp-l1r1 -T train-europarl200k/model/phrase-table.gz

This will generate a file exp-l1r1-output.xml that contains the system output

Some notes:

 * ``-T`` passes the original phrase table which will be used as a fallback option
 * ``-o`` the same output prefix used in the training step, is used as input as
   well and assumes a directory by this name exists

Evaluation
--------------

System output can subsequently be evaluated against the test set using
``colibrita-evaluate``:

    $ colibrita-evaluate --mtevaldir /path/to/mtevalscripts --ref europarl200k.test.xml --out exp-l1r1-output.xml

A summary of all Scores will be written in ``exp-l1r1-output.summary.score`` .


















