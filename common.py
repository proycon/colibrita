from __future__ import print_function, unicode_literals, division, absolute_import

from colibrita.format import SentencePair, Fragment, Writer, Reader
from pynlpl.formats.moses import PhraseTable
from pynlpl.formats.giza import GizaModel
from pycolibri import ClassDecoder, ClassEncoder, IndexedPatternModel

import sys
import os
import datetime
import subprocess
import random
import timbl

def makesentencepair(id, sourcepattern, targetpattern, sourceoffset, targetoffset, sourcesentence, targetsentence):
    targetsentence = tuple(targetsentence)
    targetpattern_n = targetpattern.count(" ") + 1
    sourcepattern_n = sourcepattern.count(" ") + 1

    newtargetsentence = tuple(targetsentence[:targetoffset]) + (Fragment(tuple(targetpattern.split())),) + tuple(targetsentence[targetoffset+targetpattern_n:])
    input = tuple(targetsentence[:targetoffset]) + (Fragment(tuple(sourcepattern.split())),) + tuple(targetsentence[targetoffset+targetpattern_n:])



    if tuple(SentencePair._str(newtargetsentence)) != targetsentence:
        print("Target sentence mismatch:\n", tuple(SentencePair._str(newtargetsentence)), "\n****VS****\n", targetsentence, file=sys.stderr)
        print("Sentence: ", id,file=sys.stderr)
        print("Source pattern: " , sourcepattern,file=sys.stderr)
        print("Target pattern: ", targetpattern,file=sys.stderr)
        print("Target offset: ", targetoffset,file=sys.stderr)
        print("Source offset: ", sourceoffset,file=sys.stderr)
        print("Target n: ", targetpattern_n,file=sys.stderr)
        print("Input: ", input,file=sys.stderr)
        return False, None

    return True, SentencePair(id, input, None, newtargetsentence)

def extractpairs(ttablefile, gizamodelfile_s2t, gizamodelfile_t2s, patternmodelfile_source, patternmodelfile_target, classfile_source, classfile_target, joinedprobabilitythreshold, divergencefrombestthreshold, DEBUG):
    if DEBUG: print("Loading phrase-table", file=sys.stderr)
    ttable = PhraseTable(ttablefile,False, False, "|||", 3, 0,None, None, lambda x: x[0] * x[2] > joinedprobabilitythreshold)

    if DEBUG: print("Loading GIZA model (s->t)", file=sys.stderr)
    gizamodel_s2t = GizaModel(gizamodelfile_s2t)
    if DEBUG: print("Loading GIZA model (t->s)", file=sys.stderr)
    gizamodel_t2s = GizaModel(gizamodelfile_t2s)

    if DEBUG: print("Loading decoders", file=sys.stderr)
    classdecoder_source = ClassDecoder(classfile_source)
    classencoder_source = ClassEncoder(classfile_source)
    classdecoder_target = ClassDecoder(classfile_target)
    classencoder_target = ClassEncoder(classfile_target)

    if DEBUG: print("Loading source pattern model " + patternmodelfile_source, file=sys.stderr)
    patternmodel_source = IndexedPatternModel(patternmodelfile_source, classencoder_source, classdecoder_source)
    if DEBUG: print("Loading target pattern model " + patternmodelfile_target, file=sys.stderr)
    patternmodel_target = IndexedPatternModel(patternmodelfile_target, classencoder_target, classdecoder_target)


    #with open(sourcecorpusfile, 'r', encoding='utf-8') as f:
    #    sourcecorpus = [x.strip() for x in f.readlines()]

    #with open(targetcorpusfile, 'r', encoding='utf-8') as f:
    #    targetcorpus = [x.strip() for x in f.readlines()]


    iter_s2t = iter(gizamodel_s2t)
    iter_t2s = iter(gizamodel_t2s)


    if DEBUG: print("Iterating over all sentence pairs", file=sys.stderr)
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
        if not intersection:
            continue

        #gather all target patterns found  in this sentence
        sourcepatterns = list(patternmodel_source.reverseindex(sentence))
        targetpatterns = [ targetpattern.decode(classdecoder_target) for targetpattern in patternmodel_target.reverseindex(sentence) ]

        if DEBUG: print("(extractpatterns) processing sentence " + str(sentence) + ", collected " + str(len(sourcepatterns)) + " source patterns and " + str(len(targetpatterns)) + " target patterns", file=sys.stderr)

        if DEBUG:
            for targetpattern in targetpatterns:
                if DEBUG: print("(extractpatterns) -- identified target pattern " + str(targetpattern) , file=sys.stderr)


        #iterate over all source patterns found in this sentence
        for sourcepattern in sourcepatterns:
            sourcepattern = sourcepattern.decode(classdecoder_source)
            if any(( ispunct(x) for x in sourcepattern.split() ) ):
                continue


            sourceindices = [ (x,y) for x,y in patternmodel_source.indices(sourcepattern) if x == sentence ]
            source_n = sourcepattern.count(" ") + 1
            assert bool(sourceindices)
            if sourcepattern in ttable:
                if DEBUG: print("(extractpatterns) -- source pattern candidate " + str(sourcepattern) + " (occuring " + str(len(sourceindices)) + " time(s)), has " + str(len(ttable[sourcepattern])) + " translation options in phrase-table" , file=sys.stderr)
                sourcesentence = s2t.source
                targetsentence = s2t.target

                targetoptions = sorted( ( (targetpattern, scores) for targetpattern, scores in ttable[sourcepattern] ) , key=lambda x: x[1] )
                bestscore = targetoptions[0][1][0] * targetoptions[0][1][2]

                #iterate over the target patterns in the phrasetable
                for targetpattern, scores in ttable[sourcepattern]:
                    if DEBUG: print("(extractpatterns) -- considering target pattern from phrase-table: " + str(targetpattern) , file=sys.stderr)
                    if targetpattern in targetpatterns:
                        if any(( ispunct(x) for x in targetpattern.split() ) ):
                            continue
                        joinedprob = scores[0] * scores[2]
                        if joinedprob < bestscore * divergencefrombestthreshold:
                            continue

                        #we have a pair, occurring in pattern models and phrase table
                        target_n = targetpattern.count(" ") + 1

                        #obtain positional offsets for source and target in sentence
                        targetindices = [ (x,y) for x,y in patternmodel_target.indices(targetpattern) if x == sentence]
                        assert bool(targetindices)

                        if DEBUG: print("(extractpatterns) --- found target pattern candidate " + str(targetpattern) + " (occuring " + str(len(targetindices)) + " time(s))" , file=sys.stderr)

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
                                    yield sourcepattern, targetpattern, sourceoffset, targetoffset, tuple(sourcesentence), tuple(targetsentence), sentence


def generate(testoutput, ttablefile, gizamodelfile_s2t, gizamodelfile_t2s, patternmodelfile_source, patternmodelfile_target, classfile_source, classfile_target, size =0, joinedprobabilitythreshold = 0.01, divergencefrombestthreshold=0.8,DEBUG = False):

    if size > 0:
        writer = Writer(testoutput+'.tmp')
    else:
        writer = Writer(testoutput)

    id = 0
    for sourcepattern, targetpattern, sourceoffset, targetoffset, sourcesentence, targetsentence, sentence in extractpairs(ttablefile, gizamodelfile_s2t, gizamodelfile_t2s, patternmodelfile_source, patternmodelfile_target, classfile_source, classfile_target, joinedprobabilitythreshold, divergencefrombestthreshold, DEBUG):
        id += 1
        valid, sentencepair = makesentencepair(id, sourcepattern, targetpattern, sourceoffset, targetoffset, sourcesentence, targetsentence)
        if valid:
            writer.write(sentencepair)

    writer.close()

    if size > 0:
        selected_ids = set(random.sample( range(1,id+1), size ))
        writer = Writer(testoutput)
        reader = Reader(testoutput+'.tmp')
        newid = 0
        for sentencepair in reader:
            if int(sentencepair.id) in selected_ids:
                newid += 1
                sentencepair.id = newid
                writer.write(sentencepair)
        reader.close()
        writer.close()


def makeset(output, settype, workdir, source, target, sourcelang, targetlang, mosesdir, bindir, size, joinedprobabilitythreshold, divergencefrombestthreshold, occurrencethreshold, debug):
    if not os.path.exists(source): # pylint: disable=E1101
        print("Source corpus " + source + " does not exist")# pylint: disable=E1101
        sys.exit(2)
    if not os.path.exists(target): # pylint: disable=E1101
        print("Target corpus " + target + " does not exist")# pylint: disable=E1101
        sys.exit(2)

    ttablefile = workdir + '/model/phrase-table.gz'
    gizamodelfile_t2s = workdir + '/giza.' + sourcelang + '-' + targetlang + '/' + sourcelang + '-' + targetlang + '.A3.final.gz'# pylint: disable=E1101
    gizamodelfile_s2t = workdir + '/giza.' + targetlang + '-' + sourcelang + '/' + targetlang + '-' + sourcelang + '.A3.final.gz'# pylint: disable=E1101
    patternmodelfile_source = workdir + '/test.' + sourcelang + '.indexedpatternmodel.colibri'# pylint: disable=E1101
    patternmodelfile_target = workdir + '/test.' + targetlang + '.indexedpatternmodel.colibri'# pylint: disable=E1101
    classfile_source = workdir + '/' + sourcelang + '.cls'# pylint: disable=E1101
    classfile_target = workdir + '/' + targetlang + '.cls'# pylint: disable=E1101

    if not os.path.exists(workdir):
        os.mkdir(workdir)
        os.chdir(workdir)
        os.symlink('../' + source, 'test.' + sourcelang)# pylint: disable=E1101
        os.symlink('../' + target, 'test.' + targetlang)# pylint: disable=E1101
    else:
        os.chdir(workdir)

    if not os.path.exists(ttablefile) or not os.path.exists(gizamodelfile_s2t) or not os.path.exists(gizamodelfile_t2s):
        if not mosesdir: print("No --mosesdir specified",file=sys.stderr)# pylint: disable=E1101
        if not buildphrasetable(settype,mosesdir, bindir, sourcelang, targetlang): return False # pylint: disable=E1101

    if not os.path.exists(patternmodelfile_source) or not os.path.exists(patternmodelfile_target) or not os.path.exists(classfile_source) or not os.path.exists(classfile_target):
        if not buildpatternmodel(settype,sourcelang, targetlang, occurrencethreshold): return False# pylint: disable=E1101

    os.chdir('..')

    if not generate(output + '.xml', ttablefile, gizamodelfile_s2t, gizamodelfile_t2s,  patternmodelfile_source, patternmodelfile_target, classfile_source, classfile_target, size, joinedprobabilitythreshold, divergencefrombestthreshold, debug): return False# pylint: disable=E1101

    return True


def buildphrasetable(corpusname, mosesdir, bindir, sourcelang, targetlang):
    EXEC_MOSES_TRAINMODEL = mosesdir + '/scripts/training/train-model.perl'
    EXTERNALBIN = bindir
    #build phrasetable using moses
    if not runcmd(EXEC_MOSES_TRAINMODEL + ' -external-bin-dir ' + EXTERNALBIN + " -root-dir . --corpus " + corpusname + " --f " + sourcelang + " --e " + targetlang + " --first-step " + str(1) + " --last-step " + str(8) + ' >&2 2> ' + corpusname + '-moses.log',"Creating word alignment and phrase table on test data", "model/phrase-table.gz"): return False
    return True

def buildpatternmodel(corpusname, sourcelang, targetlang, occurrencethreshold=2):
    options = " -t " + occurrencethreshold
    if not runcmd('classencode -o ' + sourcelang + ' ' + corpusname + '.' + sourcelang, "Encoding source corpus", sourcelang + ".cls", corpusname + '.' +  sourcelang + ".clsenc"): return False
    if not runcmd('classencode -o ' + targetlang + ' ' + corpusname + '.' + targetlang, "Encoding target corpus", targetlang + ".cls", corpusname + '.' +  targetlang + ".clsenc"): return False

    if not runcmd('patternfinder -c ' + sourcelang + '.cls -f ' + corpusname + '.' + sourcelang + '.clsenc ' + options + ' > /dev/null', "Generating pattern model for source",   corpusname + '.' +  sourcelang + ".indexedpatternmodel.colibri"): return False
    if not runcmd('patternfinder -c ' + targetlang + '.cls -f ' + corpusname + '.' + targetlang + '.clsenc ' + options + ' > /dev/null', "Generating pattern model for target",   corpusname + '.' +  targetlang + ".indexedpatternmodel.colibri"): return False

    return True

def ispunct(s):
    return s in ('.',',',':',';','/','\\','_','(',')','[',']','{','}')

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
