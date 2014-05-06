#!/usr/bin/env python3

from __future__ import print_function, unicode_literals, division, absolute_import

import argparse
import sys
import os
import subprocess
import math
from copy import copy

from pynlpl.formats.moses import PhraseTable
from colibrita.format import Reader, Writer, Fragment
from colibrita.common import extractpairs, makesentencepair, runcmd, makeset


def makebaseline(ttable, outputfile, testset,sourceencoder, targetdecoder, lm=None,tweight=1, lmweight=1):
    output = Writer(outputfile)
    for sentencepair in testset:
        print("Sentence #" + str(sentencepair.id),file=sys.stderr)
        sentencepair.ref = None
        sentencepair.output = copy(sentencepair.input)
        for left, inputfragment, right in sentencepair.inputfragments():
            inputfragment_s = str(inputfragment)
            print("\tFragment: ", inputfragment_s, file=sys.stderr)
            try:
                inputfragment_p = sourceencoder.buildpattern(inputfragment_s)
            except IOError:
                print("\tNOTICE: One or more words in '" + inputfragment_s + "' were not seen during training",file=sys.stderr)
                inputfragment_p = None

            translation = None
            if inputfragment_p in ttable:
                if lm:
                    candidatesentences = []
                    bestlmscore = -999999999
                    besttscore = -999999999
                    for targetpattern, scores in ttable[inputfragment_p]:
                        assert scores[2] >= 0 and scores[2] <= 1
                        tscore = math.log(scores[2]) #convert to base-e log (LM is converted to base-e upon load)
                        targetpattern_s = targetpattern.tostring(targetdecoder)
                        outputfragment = Fragment(tuple( targetpattern_s.split(' ') ), inputfragment.id )
                        candidatesentence = sentencepair.replacefragment(inputfragment, outputfragment, sentencepair.output)
                        lminput = " ".join(sentencepair._str(candidatesentence)).split(" ") #joining and splitting deliberately to ensure each word is one item
                        lmscore = lm.score(lminput)
                        assert lmscore <= 0
                        if lmscore > bestlmscore:
                            bestlmscore = lmscore
                        if tscore > besttscore:
                            besttscore = tscore
                        candidatesentences.append( ( candidatesentence, targetpattern, tscore, lmscore ) )
                    #get the strongest sentence
                    maxscore = -9999999999
                    for candidatesentence, targetpattern, tscore, lmscore in candidatesentences:
                        tscore = tweight * (tscore-besttscore)
                        lmscore = lmweight * (lmscore-bestlmscore)
                        score = tscore + lmscore
                        if score > maxscore:
                            maxscore = score
                            translation = targetpattern
                else:
                    maxscore = 0
                    for targetpattern, scores in ttable[inputfragment_p]:
                        targetpattern_s = targetpattern.tostring(targetdecoder)
                        if scores[2] > maxscore:
                            maxscore = scores[2]
                            translation = targetpattern_s
                outputfragment = Fragment(tuple( translation.split(' ') ), inputfragment.id )
                print("\t" + inputfragment_s + " -> " + str(outputfragment), file=sys.stderr)
                sentencepair.output = sentencepair.replacefragment(inputfragment, outputfragment, sentencepair.output)
            else:
                outputfragment = Fragment(None, inputfragment.id)
                print("\t" + inputfragment_s + " -> NO TRANSLATION", file=sys.stderr)
                sentencepair.output = sentencepair.replacefragment(inputfragment, outputfragment, sentencepair.output)
        output.write(sentencepair)
    testset.close()
    output.close()


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
    makebaseline(ttable, outputfile, testset)




if __name__ == '__main__':
    main()
    sys.exit(0)

