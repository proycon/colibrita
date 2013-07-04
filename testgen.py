#!/usr/bin/env python3

from __future__ import print_function, unicode_literals, division, absolute_import

import argparse
import sys
import os
import subprocess

from colibrita.format import Writer
from colibrita.common import extractpairs, makesentencepair, runcmd, makeset



def main():
    parser = argparse.ArgumentParser(description="Test set generation")
    parser.add_argument('--mosesdir',type=str, help="Path to moses",action='store',default="")
    parser.add_argument('--bindir',type=str, help="Path to external bin dir (path where moses bins are installed)",action='store',default="/usr/local/bin")
    parser.add_argument('--source','-s', type=str,help="Source language corpus", action='store',required=True)
    parser.add_argument('--target','-t', type=str,help="Target language corpus", action='store',required=True)
    parser.add_argument('--output','-o', type=str,help="Output name", action='store',required=True)
    parser.add_argument('--sourcelang','-S', type=str,help="Source language code (the fallback language)", action='store',required=True)
    parser.add_argument('--targetlang','-T', type=str,help="Target language code (the intended language)", action='store',required=True)
    parser.add_argument('--debug','-d', help="Debug", action='store_true', default=False)
    parser.add_argument('-p', dest='joinedprobabilitythreshold', help="Joined probabiity threshold for inclusion of fragments from phrase translation-table: min(P(s|t) * P(t|s))", type=float,action='store',default=0.01)
    parser.add_argument('-D', dest='divergencefrombestthreshold', help="Maximum divergence from best translation option", type=float,action='store',default=0.01)
    parser.add_argument('-O', dest='occurrencethreshold', help="Patterns occurring below this threshold will not be consideren", type=float,action='store',default=2)
    args = parser.parse_args()



    workdir = 'testgen-' + args.output # pylint: disable=E1101
    settype = 'test'
    makeset(args.output, settype, workdir, args.source, args.target, args.sourcelang, args.targetlang, args.mosesdir, args.bindir, args.joinedprobabilitythreshold, args.divergencefrombestthreshold, args.occurrencethreshold, args.debug)

    return True


if __name__ == '__main__':
    main()

