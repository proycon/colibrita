#!/usr/bin/env python3
from colibrita.format import Writer
from colibrita.common import extractpairs, makesentencepair

def generate(testoutput, ttablefile, gizamodelfile_s2t, gizamodelfile_t2s, patternmodelfile_source, patternmodelfile_target, classfile_source, classfile_target, DEBUG = False):
    writer = Writer(testoutput)

    id = 0
    for sourcepattern, targetpattern, sourceoffset, targetoffset, sourcesentence, targetsentence, sentence in extractpairs(ttablefile, gizamodelfile_s2t, gizamodelfile_t2s, patternmodelfile_source, patternmodelfile_target, classfile_source, classfile_target, DEBUG):
        id += 1
        makesentencepair(id, sourcepattern, targetpattern, sourceoffset, targetoffset, targetsentence)



