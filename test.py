#!/usr/bin/env python3

from __future__ import print_function, unicode_literals, division, absolute_import

import argparse
import sys
import os
import subprocess

from colibrita.format import Writer
from colibrita.common import extractpairs, makesentencepair, runcmd


def generate(testoutput, ttablefile, gizamodelfile_s2t, gizamodelfile_t2s, patternmodelfile_source, patternmodelfile_target, classfile_source, classfile_target, DEBUG = False):
    writer = Writer(testoutput)

    id = 0
    for sourcepattern, targetpattern, sourceoffset, targetoffset, sourcesentence, targetsentence, sentence in extractpairs(ttablefile, gizamodelfile_s2t, gizamodelfile_t2s, patternmodelfile_source, patternmodelfile_target, classfile_source, classfile_target, DEBUG):
        id += 1
        sentencepair = makesentencepair(id, sourcepattern, targetpattern, sourceoffset, targetoffset, targetsentence)
        writer.write(sentencepair)

def main():
    parser = argparse.ArgumentParser(description="Test set generation")
    parser.add_argument('--mosesdir',type=str, help="Path to moses",action='store')
    parser.add_argument('--source','-s', type=str,help="Source language corpus", action='store',required=True)
    parser.add_argument('--target','-t', type=str,help="Target language corpus", action='store',required=True)
    parser.add_argument('--output','-o', type=str,help="Output name", action='store',required=True)
    parser.add_argument('--sourcelang','-S', type=str,help="Source language code", action='store',required=True)
    parser.add_argument('--targetlang','-T', type=str,help="Target language code", action='store',required=True)
    parser.add_argument('--debug','-d', type=bool,help="Debug", action='store_true')
    parser.parse_args()


    if not os.path.exists(parser.source):
        print("Source corpus " + parser.source + " does not exist")
        sys.exit(2)
    if not os.path.exists(parser.target):
        print("Target corpus " + parser.target + " does not exist")
        sys.exit(2)

    testgendir = 'testgen-' + parser.output
    ttablefile = testgendir + '/model/phrase-table.gz'
    gizamodelfile_s2t = testgendir + '/giza.' + parser.sourcelang + '-' + parser.targetlang + '/' + parser.sourcelang + '-' + parser.targetlang + '.A3.final.gz'
    gizamodelfile_t2s = testgendir + '/giza.' + parser.targetlang + '-' + parser.sourcelang + '/' + parser.targetlang + '-' + parser.sourcelang + '.A3.final.gz'
    patternmodelfile_source = testgendir + '/test.' + parser.sourcelang + '.indexedpatternmodel.colibri'
    patternmodelfile_target = testgendir + '/test.' + parser.targetlang + '.indexedpatternmodel.colibri'
    classfile_source = testgendir + '/' + parser.sourcelang + '.cls'
    classfile_target = testgendir + '/' + parser.targetlang + '.cls'



    if not os.path.exists(testgendir):
        os.mkdir(testgendir)
        os.chdir(testgendir)
        os.symlink(parser.source, 'test.' + parser.sourcelang)
        os.symlink(parser.target, 'test.' + parser.targetlang)
    else:
        os.chdir(testgendir)

    if not os.path.exists(ttablefile) or not os.path.exists(gizamodelfile_s2t) or not os.path.exists(gizamodelfile_t2s):
        if not buildphrasetable(parser.mosesdir, parser.sourcelang, parser.targetlang): return False

    if not os.path.exists(patternmodelfile_source) or not os.path.exists(patternmodelfile_target) or not os.path.exists(classfile_source) or not os.path.exists(classfile_target):
        if not buildpatternmodel(parser.sourcelang, parser.targetlang): return False

    os.chdir('..')

    if not generate(parser.output + '.xml', gizamodelfile_s2t, gizamodelfile_t2s,  patternmodelfile_source, patternmodelfile_target, classfile_source, classfile_target, parser.debug): return False

    return True


def buildphrasetable(mosesdir, sourcelang, targetlang):
    EXEC_MOSES_TRAINMODEL = mosesdir + '/scripts/training/train-model.perl'
    INSTALLEDBIN = mosesdir + '/bin'
    #build phrasetable using moses
    if not runcmd(EXEC_MOSES_TRAINMODEL + ' -external-bin-dir ' + INSTALLEDBIN + " -root-dir . --corpus test --f " + sourcelang + " --e " + targetlang + " --first-step " + str(1) + " --last-step " + str(8) + ' >&2 2> testgen.log',"Creating word alignment and phrase table on test data", "model/phrase-table.gz"): return False
    return True

def buildpatternmodel(sourcelang, targetlang, options="-t 2"):
    if not runcmd('classencoder -o ' + sourcelang + ' test.' + sourcelang, "Encoding source corpus", sourcelang + ".cls", 'test.' +  sourcelang + ".clsenc"): return False
    if not runcmd('classencoder -o ' + targetlang + ' test.' + targetlang, "Encoding target corpus", targetlang + ".cls", 'test.' +  targetlang + ".clsenc"): return False

    if not runcmd('patternfinder -c ' + sourcelang + '.cls -f test.' + sourcelang + '.clsenc ' + options, "Generating pattern model for source",  'test.' +  sourcelang + ".indexedpatternmodel.colibri"): return False
    if not runcmd('patternfinder -c ' + targetlang + '.cls -f test.' + targetlang + '.clsenc ' + options, "Generating pattern model for target",  'test.' +  targetlang + ".indexedpatternmodel.colibri"): return False

    return True

if __name__ == '__main__':
    main()

