#!/usr/bin/env python3
from __future__ import print_function, unicode_literals, division, absolute_import
from colibrita.format import Writer, Reader, Fragment, Alternative
import argparse
from collections import defaultdict
import sys

def main():
    parser = argparse.ArgumentParser(description="Colibrita - Statistics on data set", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f','--dataset', type=str,help="Dataset file", action='store',default="",required=True)
    parser.add_argument('-H',dest='hist',help="Full histogram of fragments", action='store_true')
    parser.add_argument('-n',dest='nhist',help="Histogram of n-gram size", action='store_true')

    args = parser.parse_args()
    data = Reader(args.dataset)
    hist = defaultdict(int)
    nhist = defaultdict(int)
    fragcount = 0
    paircount = 0
    for sentencepair in data:
        for left, sourcefragment, right in sentencepair.inputfragments():
            if args.hist: hist[str(sourcefragment)] += 1
            if args.nhist: nhist[len(sourcefragment)] += 1
            fragcount += 1
        paircount += 1

    if args.hist:
        print("HISTOGRAM",file=sys.stderr)
        for phrase, freq in sorted(hist.items(), key=lambda x: -1 * x[1]):
            print(phrase + "\t" + str(freq))

    if args.nhist:
        print("N-HISTOGRAM",file=sys.stderr)
        for n, freq in sorted(nhist.items()):
            print(str(n) + "\t" + str(freq))

    print("Sentencepairs = ", paircount)
    print("Fragments = ", fragcount)


if __name__ == '__main__':
    main()
