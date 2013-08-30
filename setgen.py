#!/usr/bin/env python3

from __future__ import print_function, unicode_literals, division, absolute_import

import argparse
import sys
import os
import subprocess
import random

from colibrita.format import Writer
from colibrita.common import extractpairs, makesentencepair, runcmd, makeset



def main():
    parser = argparse.ArgumentParser(description="Set generation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train',dest='settype', action='store_const',const='train')
    parser.add_argument('--test',dest='settype', action='store_const',const='test')
    parser.add_argument('--mosesdir',type=str, help="Path to moses",action='store',default="")
    parser.add_argument('--bindir',type=str, help="Path to external bin dir (path where moses bins are installed)",action='store',default="/usr/local/bin")
    parser.add_argument('--source','-s', type=str,help="Source language corpus", action='store',required=True)
    parser.add_argument('--target','-t', type=str,help="Target language corpus", action='store',required=True)
    parser.add_argument('--output','-o', type=str,help="Output name", action='store',required=True)
    parser.add_argument('--sourcelang','-S', type=str,help="Source language code (the fallback language)", action='store',required=True)
    parser.add_argument('--targetlang','-T', type=str,help="Target language code (the intended language)", action='store',required=True)
    parser.add_argument('--debug','-d', help="Debug", action='store_true', default=False)
    parser.add_argument('-p', dest='joinedprobabilitythreshold', help="Joined probabiity threshold for inclusion of fragments from phrase translation-table: min(P(s|t) * P(t|s))", type=float,action='store',default=0.01)
    parser.add_argument('-D', dest='divergencefrombestthreshold', help="Maximum divergence from best translation option. If set to 0.8, the only alternatives considered are those that have a joined probability of equal or above 80\% of that the best translation option", type=float,action='store',default=0.8)
    parser.add_argument('-O', dest='occurrencethreshold', help="Patterns occurring below this threshold will not be considered", type=int,action='store',default=2)
    parser.add_argument('-n', dest='size', help="Size of set to construct (random selection, 0=maximum size)", type=int,action='store',default=0)
    parser.add_argument('--seed', help="Seed for random number generator (0=use current system time, default)", type=int,action='store',default=0)
    args = parser.parse_args()

    try:
        if args.settype != 'train' and args.settype != 'test':
            raise ValueError
    except:
        print("Specify either --train or --test")
        sys.exit(2)

    print("Building " + args.settype + " set", file=sys.stderr)
    print("Parameters: " + repr(args), file=sys.stderr)
    workdir = args.settype + '-' + args.output # pylint: disable=E1101
    if args.seed:
        random.seed(args.seed)
    makeset(args.output, args.settype, workdir, args.source, args.target, args.sourcelang, args.targetlang, args.mosesdir, args.bindir, args.size, args.joinedprobabilitythreshold, args.divergencefrombestthreshold, args.occurrencethreshold, args.debug)


if __name__ == '__main__':
    main()
    sys.exit(0)

