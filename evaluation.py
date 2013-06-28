#!/usr/bin/env python3

from __future__ import print_function, unicode_literals, division, absolute_import

import argparse
import sys
import os
import subprocess

from colibrita.format import Reader

def main():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument('--matrexdir',type=str, help="Path to Matrex evaluation scripts",action='store',default="")
    parser.add_argument('--ref',type=str,help='Reference file', action='store',required=True)
    parser.add_argument('--out',type=str,help='Output file', action='store',required=True)
    parser.add_argument('--debug','-d', help="Debug", action='store_true', default=False)
    parser.add_argument('-i',dest='casesensitive',help='Measure translation accuracy without regard for case',action='store_false',default=True)
    args = parser.parse_args()

    evaluate(Reader(args.ref), Reader(args.out), args.matrexdir)

def evaluate(ref, out, matrexdir, casesensitive=True):
    ref_it = iter(ref)
    out_it = iter(out)

    matches = 0
    misses = 0

    accuracies = []

    if casesensitive:
        eq = lambda x,y: x == y
    else:
        eq = lambda x,y: " ".join(x).lower() == " ".join(y).lower()

    while True:
        try:
            ref_s = next(ref_it)
            out_s = next(out_it)
        except StopIteration:
            break

        if ref_s.id != out_s.id:
            raise Exception("Sentence ID mismatch in reference and output! " + str(ref_s.id) + " vs " + str(out_s.id))

        outputfragments = out_s.outputfragments()
        reffragments = ref_s.reffragments()
        for inputfragment in ref_s.inputfragments().values():
            if not inputfragment.id in reffragments:
                raise Exception("No reference fragment found for fragment " + str(inputfragment.id) + " in sentence " + str(ref_s.id))

            if not inputfragment.id in outputfragments:
                print("WARNING: Input fragment " + str(inputfragment.id) + " in sentence " + str(ref_s.id) + " is not translated!", file=sys.stderr)
                misses += 1
            else:
                if eq(reffragments[inputfragment.id].value, outputfragments[inputfragment.id].value):
                    matches += 1
                else:
                    misses += 1

            accuracy = matches/(matches+misses)
            print("Accuracy for sentence " + str(id) + " = " + str(accuracy))
            accuracies.append(accuracy)

    if accuracies:
        totalavgaccuracy = sum(accuracies) / len(accuracies)
        print("Total average accuracy = " + str(totalavgaccuracy))

