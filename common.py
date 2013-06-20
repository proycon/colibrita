#!/usr/bin/env python3
from colibrita.format import Writer
from pynlpl.formats.moses import PhraseTable
from pynlpl.formats.giza import GizaModel
from pycolibri import ClassDecoder, ClassEncoder, IndexedPatternModel
import sys

def extractpairs(ttablefile, gizamodelfile_s2t, gizamodelfile_t2s, patternmodelfile_source, patternmodelfile_target, classfile_source, classfile_target):
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

        #iterate over all source patterns found in this sentence
        for sourcepattern in patternmodel_source.reverseindex(sentence):
            sourcepattern = sourcepattern.decode(classdecoder_source)
            if sourcepattern in ttable:
                #gather all target patterns found  in this sentence
                targetpatterns = list(patternmodel_target.reverseindex(sentence))

                #iterate over the target patterns in the phrasetable
                for targetpattern, scores in ttable[sourcepattern]:
                    if targetpattern in targetpatterns:
                        #we have a pair, occurring in pattern models and phrase table. now check if it doesn't violate the word alignment
                        #TODO

                        #word alignment not violated, obtain positional offset for source and target in sentence
                        #TODO

                        sourcesentence = s2t.source
                        targetsentence = s2t.target

                        #yield the pair and full context
                        yield sourcepattern, targetpattern, sourceoffset, targetoffset, sourcesentence, targetsentence, sentence


