#!/usr/bin/env python3
from colibrita.format import Writer
from pynlpl.formats.moses import PhraseTable
from pynlpl.formats.giza import GizaModel
from pycolibri import ClassDecoder, ClassEncoder, IndexedPatternModel
import sys

def extractpairs(ttablefile, gizamodelfile_s2t, gizamodelfile_t2s, patternmodelfile_source, patternmodelfile_target, classfile_source, classfile_target, DEBUG = False):
    ttable = PhraseTable(ttablefile)

    gizamodel_s2t = GizaModel(gizamodelfile_s2t)
    gizamodel_t2s = GizaModel(gizamodelfile_t2s)

    classdecoder_source = ClassDecoder(classfile_source)
    classencoder_source = ClassEncoder(classfile_source)
    classdecoder_target = ClassDecoder(classfile_target)
    classencoder_target = ClassEncoder(classfile_target)

    patternmodel_source = IndexedPatternModel(patternmodelfile_source, classencoder_source, classdecoder_source)
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
            s2t = iter_s2t.next()
            t2s = iter_t2s.next()
        except StopIteration:
            print("WARNING: No more GIZA alignments, breaking",file=sys.stderr)
            break

        sentence = s2t.index
        assert t2s.index == s2t.index

        intersection = s2t.intersect(t2s)


        #gather all target patterns found  in this sentence
        sourcepatterns = list(patternmodel_source.reverseindex(sentence))
        targetpatterns = list(patternmodel_target.reverseindex(sentence))

        if DEBUG: print("(extractpatterns) processing sentence " + str(sentence) + ", collected " + str(len(sourcepatterns)) + " source patterns and " + str(len(targetpatterns)) + " target patterns", file=sys.stderr)

        #iterate over all source patterns found in this sentence
        for sourcepattern in sourcepatterns:
            sourcepattern = sourcepattern.decode(classdecoder_source)
            sourceindices = list(patternmodel_source.indices(sourcepattern))
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

