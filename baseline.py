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

