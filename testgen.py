#!/usr/bin/env python3

from __future__ import print_function, unicode_literals, division, absolute_import

import argparse
import sys
import os
import subprocess

from colibrita.format import Writer
from colibrita.common import extractpairs, makesentencepair, runcmd


def generate(testoutput, ttablefile, gizamodelfile_s2t, gizamodelfile_t2s, patternmodelfile_source, patternmodelfile_target, classfile_source, classfile_target,  joinedprobabilitythreshold = 0.01, divergencefrombestthreshold=0.8,DEBUG = False):

    writer = Writer(testoutput)

    id = 0
    for sourcepattern, targetpattern, sourceoffset, targetoffset, sourcesentence, targetsentence, sentence in extractpairs(ttablefile, gizamodelfile_s2t, gizamodelfile_t2s, patternmodelfile_source, patternmodelfile_target, classfile_source, classfile_target, joinedprobabilitythreshold, divergencefrombestthreshold, DEBUG):
        id += 1
        sentencepair = makesentencepair(id, sourcepattern, targetpattern, sourceoffset, targetoffset, sourcesentence, targetsentence)
        writer.write(sentencepair)

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
    args = parser.parse_args()


    if not os.path.exists(args.source): # pylint: disable=E1101
        print("Source corpus " + args.source + " does not exist")# pylint: disable=E1101
        sys.exit(2)
    if not os.path.exists(args.target): # pylint: disable=E1101
        print("Target corpus " + args.target + " does not exist")# pylint: disable=E1101
        sys.exit(2)

    testgendir = 'testgen-' + args.output # pylint: disable=E1101
    ttablefile = testgendir + '/model/phrase-table.gz'
    gizamodelfile_s2t = testgendir + '/giza.' + args.sourcelang + '-' + args.targetlang + '/' + args.sourcelang + '-' + args.targetlang + '.A3.final.gz'# pylint: disable=E1101
    gizamodelfile_t2s = testgendir + '/giza.' + args.targetlang + '-' + args.sourcelang + '/' + args.targetlang + '-' + args.sourcelang + '.A3.final.gz'# pylint: disable=E1101
    patternmodelfile_source = testgendir + '/test.' + args.sourcelang + '.indexedpatternmodel.colibri'# pylint: disable=E1101
    patternmodelfile_target = testgendir + '/test.' + args.targetlang + '.indexedpatternmodel.colibri'# pylint: disable=E1101
    classfile_source = testgendir + '/' + args.sourcelang + '.cls'# pylint: disable=E1101
    classfile_target = testgendir + '/' + args.targetlang + '.cls'# pylint: disable=E1101



    if not os.path.exists(testgendir):
        os.mkdir(testgendir)
        os.chdir(testgendir)
        os.symlink('../' + args.source, 'test.' + args.sourcelang)# pylint: disable=E1101
        os.symlink('../' + args.target, 'test.' + args.targetlang)# pylint: disable=E1101
    else:
        os.chdir(testgendir)

    if not os.path.exists(ttablefile) or not os.path.exists(gizamodelfile_s2t) or not os.path.exists(gizamodelfile_t2s):
        if not args.mosesdir: print("No --mosesdir specified",file=sys.stderr)# pylint: disable=E1101
        if not buildphrasetable(args.mosesdir, args.bindir, args.sourcelang, args.targetlang): return False # pylint: disable=E1101

    if not os.path.exists(patternmodelfile_source) or not os.path.exists(patternmodelfile_target) or not os.path.exists(classfile_source) or not os.path.exists(classfile_target):
        if not buildpatternmodel(args.sourcelang, args.targetlang): return False# pylint: disable=E1101

    os.chdir('..')

    if not generate(args.output + '.xml', ttablefile, gizamodelfile_s2t, gizamodelfile_t2s,  patternmodelfile_source, patternmodelfile_target, classfile_source, classfile_target, args.joinedprobabilitythreshold, args.divergencefrombestthreshold, args.debug): return False# pylint: disable=E1101

    return True


def buildphrasetable(mosesdir, bindir, sourcelang, targetlang):
    EXEC_MOSES_TRAINMODEL = mosesdir + '/scripts/training/train-model.perl'
    EXTERNALBIN = bindir
    #build phrasetable using moses
    if not runcmd(EXEC_MOSES_TRAINMODEL + ' -external-bin-dir ' + EXTERNALBIN + " -root-dir . --corpus test --f " + sourcelang + " --e " + targetlang + " --first-step " + str(1) + " --last-step " + str(8) + ' >&2 2> testgen.log',"Creating word alignment and phrase table on test data", "model/phrase-table.gz"): return False
    return True

def buildpatternmodel(sourcelang, targetlang, options="-t 2"):
    if not runcmd('classencode -o ' + sourcelang + ' test.' + sourcelang, "Encoding source corpus", sourcelang + ".cls", 'test.' +  sourcelang + ".clsenc"): return False
    if not runcmd('classencode -o ' + targetlang + ' test.' + targetlang, "Encoding target corpus", targetlang + ".cls", 'test.' +  targetlang + ".clsenc"): return False

    if not runcmd('patternfinder -c ' + sourcelang + '.cls -f test.' + sourcelang + '.clsenc ' + options + ' > /dev/null', "Generating pattern model for source",  'test.' +  sourcelang + ".indexedpatternmodel.colibri"): return False
    if not runcmd('patternfinder -c ' + targetlang + '.cls -f test.' + targetlang + '.clsenc ' + options + ' > /dev/null', "Generating pattern model for target",  'test.' +  targetlang + ".indexedpatternmodel.colibri"): return False

    return True

if __name__ == '__main__':
    main()

