#!/usr/bin/env python3

from __future__ import print_function, unicode_literals, division, absolute_import

import argparse
import sys
import os
import subprocess
from colibrita.format import Reader, Writer, Fragment


def main():
    parser = argparse.ArgumentParser(description="Colibrita - Translation Assistance using Moses", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f','--dataset', type=str,help="Dataset file", action='store',default="",required=False)
    parser.add_argument('--debug','-d', help="Debug", action='store_true', default=False)
    parser.add_argument('-o','--output',type=str,help="Output prefix", required = True)
    parser.add_argument('-T','--ttable', type=str,help="Phrase translation table (file) to use when testing with --lm and without classifier training", action='store',required=True)
    parser.add_argument('--lm',type=str, help="Language model (file in ARPA format, as produced by for instance SRILM)", action='store',required=True)
    parser.add_argument('--lmweight',type=float, help="Language model weight", action='store',default=0.5)
    parser.add_argument('--lmorder',type=float, help="Language model order", action='store',default=3)
    parser.add_argument('--dweight',type=float, help="Distortion weight", action='store',default=0.6)
    parser.add_argument('--tmweights',type=str, help="Translation model weights (comma separated)", action='store',default="0.20,0.20,0.20,0.20,0.20")

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

    print("Calling moses: moses -f " + args.output + '.moses.ini' ,file=sys.stderr)
    p = subprocess.Popen('moses -f ' + args.output + '.moses.ini',shell=True,stdout=subprocess.PIPE,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    for sentencepair in data:
        for left, sourcefragment, right in sentencepair.inputfragments():
            p.stdin.write( (str(sourcefragment) + "\n").encode('utf-8'))
    solutions = p.communicate()[0].split(b"\n")
    p.stdin.close()

    data.reset()

    print("Processing moses output...",file=sys.stderr)
    writer = Writer(args.output + '.output.xml')

    solutionindex = 0
    for sentencepair in data:
        for left, inputfragment, right in sentencepair.inputfragments():
            solution = solutions[solutionindex]
            if solution[-1] == '.': solution = solution[:-1]
            outputfragment = Fragment(tuple(str(solution,'utf-8').split()), inputfragment.id)
            print("\t" + str(inputfragment) + " -> " + str(outputfragment), file=sys.stderr)
            sentencepair.output = sentencepair.ref
            sentencepair.output = sentencepair.replacefragment(inputfragment, outputfragment, sentencepair.output)
            sentencepair.ref = None
            solutionindex += 1
            writer.write(sentencepair)
    writer.close()

    print("All done.", file=sys.stderr)


if __name__ == '__main__':
    main()
