#!/usr/bin/env python3
from colibrita.format import Writer, SentencePair, Fragment
from colibrita.common import extractpairs


def maketestsentencepair(id, sourcepattern, targetpattern, sourceoffset, targetoffset, targetsentence):
    targetsentence = tuple(targetsentence.split())
    targetpattern_n = targetpattern.count(" ") + 1

    input = tuple(targetsentence[:targetoffset]) + tuple(Fragment(sourcepattern)) + tuple(targetsentence[targetoffset+targetpattern_n:])

    SentencePair(id, input, None, targetsentence)


def generate(testoutput, ttablefile, gizamodelfile_s2t, gizamodelfile_t2s, patternmodelfile_source, patternmodelfile_target, classfile_source, classfile_target, DEBUG = False):
    writer = Writer(testoutput)

    id = 0
    for sourcepattern, targetpattern, sourceoffset, targetoffset, sourcesentence, targetsentence, sentence in extractpairs(ttablefile, gizamodelfile_s2t, gizamodelfile_t2s, patternmodelfile_source, patternmodelfile_target, classfile_source, classfile_target, DEBUG):
        id += 1
        maketestsentencepair(id, sourcepattern, targetpattern, sourceoffset, targetoffset, targetsentence)



