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
---------------

Colibrita is written in Python 3.

First make sure you have a modern linux distribution with the necessary prerequisites: python3-dev, python3-setuptools, python3-lxml, cython3, gcc, g++,  autoconf,  automake, autoconf-archive, libtool 

Other unix systems including FreeBSD and Mac OS X will most likely work fine too, but some extra effort may be
required in installing things. The instructions here have been tailored for
Debian/Ubuntu-based Linux distributions.

Colibrita depends on pynlpl and colibri-core. Install pynlpl:

    $ sudo easy_install3 pynlpl

Download colibri-core from https://github.com/proycon/colibri-core and install as follows:

    $ cd colibri-core
    $ bash bootstrap
    $ ./configure 
    $ make
    $ sudo make install
    $ sudo python3 ./setup.py build_ext 
    $ sudo python3 ./setup.py install

Then install colibrita:

    $ cd colibrita
    $ sudo python3 ./setup.py install

Usage
-------------

    






