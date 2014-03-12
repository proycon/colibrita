#!/usr/bin/env python3

from __future__ import print_function, unicode_literals, division, absolute_import

import argparse
import sys
import os
import subprocess
from colibrita.format import Reader, Writer, Fragment, Alternative
from pynlpl.lm.lm import ARPALanguageModel
from copy import copy
import math


def main():
    parser = argparse.ArgumentParser(description="Colibrita - Translation Assistance using Moses", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f','--dataset', type=str,help="Dataset file", action='store',default="",required=True)
    parser.add_argument('--debug','-d', help="Debug", action='store_true', default=False)
    parser.add_argument('-o','--output',type=str,help="Output prefix", required = True)
    parser.add_argument('-T','--ttable', type=str,help="Phrase translation table (file) to use when testing with --lm and without classifier training", action='store',required=True)
    parser.add_argument('--lm',type=str, help="Language model (file in ARPA format, as produced by for instance SRILM)", action='store',required=True)
    parser.add_argument('--lmweight',type=float, help="Language model weight for Moses ", action='store',default=0.5)
    parser.add_argument('--lmorder',type=float, help="Language model order", action='store',default=3)
    parser.add_argument('--dweight',type=float, help="Distortion weight for Moses", action='store',default=0.6)
    parser.add_argument('--tmweights',type=str, help="Translation model weights for Moses (comma separated)", action='store',default="0.20,0.20,0.20,0.20,0.20")
    parser.add_argument('--lmweightrr',type=float, help="Language model weight in reranking", action='store',default=1)
    parser.add_argument('--tweightrr',type=float, help="Translation model weight in reranking", action='store',default=1)
    parser.add_argument('-n','--n',type=int,help="Number of output hypotheses per sentence", default=25)
    parser.add_argument('-a','--a',type=int,help="Add alternative translations, up to the specified numer", default=0)

    args = parser.parse_args()

    #if os.path.exists(args.output):
    #    print("Output already " + args.output + " already exists, doing nothing..",file=sys.stderr)
    #    sys.exit(2)
    #else:
    #    os.mkdir(args.output)

    if not os.path.exists(args.ttable):
        print("Translation table " + args.ttable + " does not exist", file=sys.stderr)
        sys.exit(2)

    if not os.path.exists(args.lm):
        print("Language model " + args.lm + " does not exist", file=sys.stderr)
        sys.exit(2)


    data = Reader(args.dataset)

    f = open(args.output + '.moses.ini','w',encoding='utf-8')
    f.write("[input-factors]\n0\n\n")
    f.write("[mapping]\n0 T 0\n\n")
    f.write("[ttable-file]\n0 0 0 5 " + args.ttable + "\n\n")
    f.write("[lmodel-file]\n0 0 " + str(args.lmorder) + " " + args.lm + "\n\n")
    f.write("[ttable-limit]\n20\n\n")
    f.write("[weight-d]\n" + str(args.dweight) + "\n\n")
    f.write("[weight-l]\n" + str(args.lmweight) + "\n\n")
    f.write("[weight-t]\n" + "\n".join(args.tmweights.split(',')) + "\n\n")
    f.write("[weight-w]\n-1\n")
    f.write("[distortion-limit]\n6\n")
    f.close()


    if not os.path.exists(args.output + ".nbestlist"):
        cmd = 'moses -f ' + args.output + '.moses.ini -n-best-list ' + args.output + '.nbestlist ' + str(args.n)
        print("Calling moses: " + cmd,file=sys.stderr)
        p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        for sentencepair in data:
            for left, sourcefragment, right in sentencepair.inputfragments():
                p.stdin.write( (str(sourcefragment) + "\n").encode('utf-8'))
        p.communicate()
        p.stdin.close()

        data.reset()
    else:
        print("Moses output already exists, not overwriting. Delete " + args.output + ".nbestlist if you want a fresh run.",file=sys.stderr)


    print("Loading Language model", file=sys.stderr)
    lm = ARPALanguageModel(args.lm)

    print("Processing moses output...",file=sys.stderr)

    previndex = -1
    sentenceoutput = []
    hypotheses = []
    with open(args.output+'.nbestlist','r',encoding='utf-8') as f:
        for line in f:
            fields = [ x.strip() for x in  line.strip().split("|||") ]
            print(fields,file=sys.stderr)
            index = int(fields[0])
            if index != previndex:
                if hypotheses:
                    sentenceoutput.append( hypotheses )
                hypotheses = []
            previndex = index
            solution = fields[1]
            rawscores = fields[2].split(' ')
            print(rawscores,file=sys.stderr)
            tscore = float(rawscores[9])
            hypotheses.append( (solution, tscore) )
        sentenceoutput.append( hypotheses ) #don't forget last one

    writer = Writer(args.output + '.output.xml')
    for i, sentencepair in enumerate(data):
        sentencepair.output = copy(sentencepair.input)
        hypotheses = sentenceoutput[i]
        for left, inputfragment, right in sentencepair.inputfragments():
            candidatesentences = []
            bestlmscore = -999999999
            besttscore = -999999999
            for hypothesis, tscore in hypotheses:
                #compute new lm score
                outputfragment = Fragment(tuple(hypothesis.split(' ')), inputfragment.id)
                candidatesentence = sentencepair.replacefragment(inputfragment, outputfragment, sentencepair.output)
                lminput = " ".join(sentencepair._str(candidatesentence)).split(" ") #joining and splitting deliberately to ensure each word is one item
                lmscore = lm.score(lminput)
                assert lmscore <= 0
                if lmscore > bestlmscore:
                    bestlmscore = lmscore
                if tscore > besttscore:
                    besttscore = tscore

                candidatesentences.append( ( candidatesentence, hypothesis, tscore, lmscore ) )

            #compute scores
            solutions = []
            for candidatesentence, targetpattern, tscore, lmscore in candidatesentences:
                tscore = args.tweightrr * (tscore-besttscore)
                lmscore = args.lmweightrr * (lmscore-bestlmscore)
                score = tscore + lmscore
                print(targetpattern + " --- tscore=" + str(tscore) + ", lmscore=" + str(lmscore),file=sys.stderr)
                solutions.append( (score, targetpattern) )

            solutions = sorted(solutions, key=lambda x: -1 * x[0])

            translation = tuple(solutions[0][1].split())
            outputfragment = Fragment(translation, inputfragment.id)
            print("\t" + str(inputfragment) + " -> " + str(outputfragment), file=sys.stderr)

            if args.a:
                for score, solution in solutions[1:1+args.a]:
                    outputfragment.alternatives.append( Alternative( tuple(solution.split()), confidence=score) )

            sentencepair.output = sentencepair.replacefragment(inputfragment, outputfragment, sentencepair.output)

            writer.write(sentencepair)

            break #only support one iteration for now, one fragment per sentence
    writer.close()


    print("All done.", file=sys.stderr)


if __name__ == '__main__':
    main()
