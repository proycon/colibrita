from __future__ import print_function, unicode_literals, division, absolute_import

from colibrita.format import SentencePair, Fragment
from pynlpl.formats.moses import PhraseTable
from pynlpl.formats.giza import GizaModel
from pycolibri import ClassDecoder, ClassEncoder, IndexedPatternModel

import sys
import os
import datetime
import subprocess

def makesentencepair(id, sourcepattern, targetpattern, sourceoffset, targetoffset, targetsentence):
    targetsentence = tuple(targetsentence.split())
    targetpattern_n = targetpattern.count(" ") + 1

    input = tuple(targetsentence[:targetoffset]) + tuple(Fragment(sourcepattern)) + tuple(targetsentence[targetoffset+targetpattern_n:])

    return SentencePair(id, input, None, targetsentence)

def extractpairs(ttablefile, gizamodelfile_s2t, gizamodelfile_t2s, patternmodelfile_source, patternmodelfile_target, classfile_source, classfile_target, DEBUG = False):
    if DEBUG: print("Loading phrase-table", file=sys.stderr)
    ttable = PhraseTable(ttablefile)

    if DEBUG: print("Loading GIZA model (s->t)", file=sys.stderr)
    gizamodel_s2t = GizaModel(gizamodelfile_s2t)
    if DEBUG: print("Loading GIZA model (t->s)", file=sys.stderr)
    gizamodel_t2s = GizaModel(gizamodelfile_t2s)

    if DEBUG: print("Loading decoders", file=sys.stderr)
    classdecoder_source = ClassDecoder(classfile_source)
    classencoder_source = ClassEncoder(classfile_source)
    classdecoder_target = ClassDecoder(classfile_target)
    classencoder_target = ClassEncoder(classfile_target)

    if DEBUG: print("Loading source pattern model", file=sys.stderr)
    patternmodel_source = IndexedPatternModel(patternmodelfile_source, classencoder_source, classdecoder_source)
    if DEBUG: print("Loading target pattern model", file=sys.stderr)
    patternmodel_target = IndexedPatternModel(patternmodelfile_target, classencoder_target, classdecoder_target)


    #with open(sourcecorpusfile, 'r', encoding='utf-8') as f:
    #    sourcecorpus = [x.strip() for x in f.readlines()]

    #with open(targetcorpusfile, 'r', encoding='utf-8') as f:
    #    targetcorpus = [x.strip() for x in f.readlines()]


    iter_s2t = iter(gizamodel_s2t)
    iter_t2s = iter(gizamodel_t2s)


    #iterate over all sentences in the parallel corpus (GIZA alignment acts as source)
    while True:
        try:
            s2t = next(iter_s2t)
            t2s = next(iter_t2s)
        except StopIteration:
            print("WARNING: No more GIZA alignments, breaking",file=sys.stderr)
            break

        sentence = s2t.index
        assert t2s.index == s2t.index

        if DEBUG:
            print("(extractpatterns) s2t.source=", s2t.source , file=sys.stderr)
            print("(extractpatterns) t2s.target=", t2s.target , file=sys.stderr)
            print("(extractpatterns) t2s.source=", t2s.source , file=sys.stderr)
            print("(extractpatterns) s2t.target=", s2t.target , file=sys.stderr)
        intersection = s2t.intersect(t2s)


        #gather all target patterns found  in this sentence
        sourcepatterns = list(patternmodel_source.reverseindex(sentence))
        targetpatterns = list(patternmodel_target.reverseindex(sentence))

        if DEBUG: print("(extractpatterns) processing sentence " + str(sentence) + ", collected " + str(len(sourcepatterns)) + " source patterns and " + str(len(targetpatterns)) + " target patterns", file=sys.stderr)



        #iterate over all source patterns found in this sentence
        for sourcepattern in sourcepatterns:
            if DEBUG: print("1", file=sys.stderr)
            print(len(sourcepattern))
            if DEBUG: print("2", file=sys.stderr)
            sourcepattern = sourcepattern.decode(classdecoder_source)
            if DEBUG: print("3", file=sys.stderr)
            sourceindices = list(patternmodel_source.indices(sourcepattern))
            if DEBUG: print("4", file=sys.stderr)
            source_n = sourcepattern.count(" ") + 1
            assert bool(sourceindices)
            if sourcepattern in ttable:
                if DEBUG: print("(extractpatterns) -- source pattern candidate " + str(sourcepattern) + " (occuring " + len(sourceindices) + " time(s))" , file=sys.stderr)
                sourcesentence = s2t.source
                targetsentence = s2t.target


                #iterate over the target patterns in the phrasetable
                for targetpattern, scores in ttable[sourcepattern]:
                    if targetpattern in targetpatterns:


                        #we have a pair, occurring in pattern models and phrase table
                        target_n = targetpattern.count(" ") + 1

                        #obtain positional offsets for source and target in sentence
                        targetindices = list(patternmodel_target.indices(targetpattern))
                        assert bool(targetindices)

                        if DEBUG: print("(extractpatterns) --- found target pattern candidate " + str(targetpattern) + " (occuring " + len(targetindices) + " time(s))" , file=sys.stderr)

                        #yield the pair and full context
                        for _, sourceoffset in sourceindices:
                            for _, targetoffset in targetindices:
                                #check if offsets don't violate the word alignment
                                valid = True
                                for i in range(sourceoffset, sourceoffset + source_n):
                                    target, foundindex = intersection.getalignedtarget(i)
                                    if isinstance(foundindex, tuple):
                                        targetl = foundindex[1]
                                        foundindex = foundindex[0]
                                    if foundindex < targetoffset or foundindex >= targetoffset + target_n:
                                        valid = False
                                        if DEBUG: print("(extractpatterns) --- violates word alignment", file=sys.stderr)
                                        break
                                if valid:
                                    if DEBUG: print("(extractpatterns) --- ok", file=sys.stderr)
                                    yield sourcepattern, targetpattern, sourceoffset, targetoffset, sourcesentence, targetsentence, sentence




#move to seperate module?

def bold(s):
    CSI="\x1B["
    return CSI+"1m" + s + CSI + "0m"

def white(s):
    CSI="\x1B["
    return CSI+"37m" + s + CSI + "0m"


def red(s):
    CSI="\x1B["
    return CSI+"31m" + s + CSI + "0m"

def green(s):
    CSI="\x1B["
    return CSI+"32m" + s + CSI + "0m"


def yellow(s):
    CSI="\x1B["
    return CSI+"33m" + s + CSI + "0m"


def blue(s):
    CSI="\x1B["
    return CSI+"34m" + s + CSI + "0m"


def magenta(s):
    CSI="\x1B["
    return CSI+"35m" + s + CSI + "0m"


def log(msg, color=None, dobold = False):
    if color:
        msg = color(msg)
    if dobold:
        msg = bold(msg)
    print(msg, file=sys.stderr)

def execheader(name,*outputfiles, **kwargs):
    print("----------------------------------------------------",file=sys.stderr)
    if outputfiles:
        skip = True
        for outputfile in outputfiles:
            if not os.path.exists(outputfile):
                skip = False
                break
        if skip:
            log("Skipping " + name, yellow, True)
            return False
    if 'cmd' in kwargs:
        log("Calling " + name + " " + timestamp() ,white, True)
        log("Command "+ ": " + kwargs['cmd'])
    else:
        log("Calling " + name + " " + timestamp(),white, True)
    return True

def timestamp():
    return "\t" + magenta("@" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

def execfooter(name, r, *outputfiles, **kwargs):
    if 'successcodes' in kwargs:
        successcodes = kwargs['successcodes']
    else:
        successcodes = [0]
    if r in successcodes:
        log("Finished " + name + " " + timestamp(),green,True)
    else:
        log("Runtime error from " + name + ' (return code ' + str(r) + ') ' + timestamp(),red,True)
        return False
    if outputfiles:
        error = False
        for outputfile in outputfiles:
            if os.path.exists(outputfile):
                log("Produced output file " + outputfile,green)
            else:
                log("Expected output file " + outputfile+ ", not produced!",red)
                error = True
        if error:
            return False
    return True

def runcmd(cmd, name, *outputfiles, **kwargs):
    if not execheader(name,*outputfiles, cmd=cmd): return True
    r = subprocess.call(cmd, shell=True)
    return execfooter(name, r, *outputfiles,**kwargs)
