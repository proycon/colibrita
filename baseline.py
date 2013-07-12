#!/usr/bin/env python3

from __future__ import print_function, unicode_literals, division, absolute_import

import argparse
import sys
import os
import subprocess
from copy import copy

from pynlpl.formats.moses import PhraseTable
from colibrita.format import Reader, Writer, Fragment
from colibrita.common import extractpairs, makesentencepair, runcmd, makeset


def main():
    parser = argparse.ArgumentParser(description="Baseline Translation Assistance System, performs simple phrase-table lookup and substition without context information, reordering, or language modelling")
    parser.add_argument('-t','--testset', type=str,help="Testset file", action='store',required=True)
    parser.add_argument('-o','--output', type=str,help="Output filename", action='store',default='baseline')
    parser.add_argument('-T','--ttable', type=str,help="Phrase translation table (file)", action='store',required=True)
    parser.add_argument('--debug','-d', help="Debug", action='store_true', default=False)
    args = parser.parse_args()


    ttable = PhraseTable(args.ttable,False, False, "|||", 3, 0,None, None)

    outputfile = args.output
    if outputfile.lower()[-4:] != '.xml':
        outputfile += '.xml'
    testset = Reader(args.testset)
    output = Writer(outputfile)
    for sentencepair in testset:
        print("Sentence #" + sentencepair.id,file=sys.stderr)
        sentencepair.ref = None
        sentencepair.output = copy(sentencepair.input)
        for left, inputfragment, right in sentencepair.inputfragments():
            translation = None
            if str(inputfragment) in ttable:
                maxscore = 0
                for targetpattern, scores in ttable[str(inputfragment)]:
                    if scores[2] > maxscore:
                        maxscore = scores[2]
                        translation = targetpattern
                translation = tuple(translation.split())
                outputfragment = Fragment(translation, inputfragment.id)
                print("\t" + str(inputfragment) + " -> " + str(outputfragment), file=sys.stderr)
            else:
                outputfragment = Fragment(None, inputfragment.id)
                print("\t" + str(inputfragment) + " -> NO TRANSLATION", file=sys.stderr)
            sentencepair.output = sentencepair.replacefragment(inputfragment, outputfragment, sentencepair.output)
        output.write(sentencepair)
    testset.close()
    output.close()


    return True


if __name__ == '__main__':
    main()

