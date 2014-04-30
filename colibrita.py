#!/usr/bin/env python3

from __future__ import print_function, unicode_literals, division, absolute_import

import argparse
import sys
import os
import itertools
import glob
import math
import timbl
import datetime
import pickle
from collections import defaultdict
from urllib.parse import quote_plus, unquote_plus
from copy import copy

from colibrita.format import Writer, Reader, Fragment, Alternative
from colibrita.baseline import makebaseline
from colibricore import ClassEncoder, ClassDecoder, PatternSetModel
from colibrimt.alignmentmodel import AlignmentModel
from pynlpl.lm.lm import ARPALanguageModel

try:
    from twisted.web import server, resource
    from twisted.internet import reactor

    class ColibritaProcessorResource(resource.Resource):
        isLeaf = True
        numberRequests = 0

        def __init__(self, experts, dttable, ttable, lm, args, timbloptions):
            self.experts = experts
            self.dttable = dttable
            self.ttable = ttable
            self.lm = lm
            self.args = args
            self.timbloptions = timbloptions

        #def render_GET(self, request):
        #    self.numberRequests += 1
        #    if b'input' in request.args:
        #        request.setHeader(b"content-type", b"application/xml")
        #        print("Server input: ", request.args[b'input'][0], file=sys.stderr)
        #        line = str(request.args[b'input'][0],'utf-8')
        #        sentencepair = plaintext2sentencepair(line)
        #        if self.experts:
        #            sentencepair = self.experts.processsentence(sentencepair, self.dttable, self.args.leftcontext, self.args.rightcontext, self.args.keywords, self.timbloptions, self.lm, self.args.tmweight, self.args.lmweight)
        #        elif self.ttable:
        #            pass #TODO
        #        return lxml.etree.tostring(sentencepair.xml(), encoding='utf-8',xml_declaration=False, pretty_print=True)
        #    else:
        #        request.setHeader(b"content-type", b"text/html")
        #        return b"""<?xml version="1.0" encoding="utf-8"?>
#<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
#<html xmlns="http://www.w3.org/1999/xhtml" lang="en">
#<html>
#  <head>
#        <meta http-equiv="content-type" content="application/xhtml+xml; charset=utf-8"/>
#        <title>Colibrita &cdot; Translation Assistant</title>
#  </head>
#  <body>
#      Enter text in target language, enclose fall-back language content in asteriskes (*):<br />
#      <form action="/" method="get">
#          <input name="input" /><br />
#          <input type="submit">
#      </form>
#  </body>
#</html>"""

    class ColibritaServer:
        def __init__(self, port, experts, dttable, ttable, lm, args, timbloptions):
            assert isinstance(port, int)
            reactor.listenTCP(port, server.Site(ColibritaProcessorResource(experts,dttable, ttable,lm, args, timbloptions)))
            reactor.run()

except ImportError:
    print("(Webserver support not available)",file=sys.stderr)


class NoTranslationException(Exception):
    pass

MAXKEYWORDS = 50

class ClassifierExperts:
    def __init__(self, workdir):
        self.workdir = workdir
        self.classifiers = {}
        self.keywords = {}

    def load(self, timbloptions, leftcontext, rightcontext,dokeywords, limit=None, autoconf=False):
        for f in glob.glob(self.workdir + '/*.train'):
            sourcefragment = unquote_plus(os.path.basename(f).replace('.train',''))
            if limit and not sourcefragment in limit:
                print("NOTICE: Classifier '" + sourcefragment + "' not found in testset, skipping...", file=sys.stderr)
                continue
            print("Loading classifier " + sourcefragment, file=sys.stderr)
            self.classifiers[sourcefragment] = timbl.TimblClassifier(f[:-6], timbloptions)
            self.classifiers[sourcefragment].leftcontext = None
            self.classifiers[sourcefragment].rightcontext = None
            self.classifiers[sourcefragment].keywords = None
            self.classifiers[sourcefragment].selectedleftcontext = None
            self.classifiers[sourcefragment].selectedrightcontext = None
            self.classifiers[sourcefragment].selectedkeywords = None
            kwfile = f.replace('.train','.keywords')
            if os.path.exists(kwfile):
                if sourcefragment in self.classifiers:
                    self.classifiers[sourcefragment].keywords = True
                self.keywords[sourcefragment] = []
                print("Loading keywords for " + sourcefragment, file=sys.stderr)
                kwf = open(kwfile, 'r', encoding='utf-8')
                for line in kwf:
                    keyword, target, c, p = line.split("\t")
                    c = int(c)
                    p = float(p)
                    self.keywords[sourcefragment].append((keyword, target,c,p))
                kwf.close()
                self.keywords[sourcefragment] = sorted(self.keywords[sourcefragment], key=lambda x: -1 * x[3])



            conffile = f.replace('.train','.conf')
            if os.path.exists(conffile):
                configid, bestconfigid, timblopts, accuracy = self.readconf(sourcefragment)
                if configid:
                    #warning supports only one-digit contexts
                    l = configid.find('l')
                    r = configid.find('r')
                    self.classifiers[sourcefragment].leftcontext = int(configid[l+1:l+2])
                    self.classifiers[sourcefragment].rightcontext = int(configid[r+1:r+2])
                    self.classifiers[sourcefragment].keywords = sourcefragment in self.keywords
                if bestconfigid:
                    l = bestconfigid.find('l')
                    r = bestconfigid.find('r')
                    self.classifiers[sourcefragment].selectedleftcontext = int(bestconfigid[l+1:l+2])
                    self.classifiers[sourcefragment].selectedrightcontext = int(bestconfigid[r+1:r+2])
                    self.classifiers[sourcefragment].selectedkeywords = bestconfigid[-1] == 'k'

                if autoconf and timblopts:
                    self.classifiers[sourcefragment].timbloptions += ' ' + timblopts
                elif not autoconf and (self.classifiers[sourcefragment].leftcontext != leftcontext or self.classifiers[sourcefragment].rightcontext != rightcontext or self.classifiers[sourcefragment].keywords != dokeywords):
                    assert leftcontext <= self.classifiers[sourcefragment].leftcontext
                    assert rightcontext <= self.classifiers[sourcefragment].rightcontext
                    self.classifiers[sourcefragment].timbloptions += ' '  + self.gettimblskipopts(sourcefragment, self.classifiers[sourcefragment].leftcontext,self.classifiers[sourcefragment].rightcontext,leftcontext,rightcontext, ((sourcefragment in self.keywords) and not dokeywords) )


                print(" \- Loaded configuration " + configid + ", timbloptions: ", self.classifiers[sourcefragment].timbloptions , file=sys.stderr)
        print("Loaded " + str(len(self.classifiers)) + " classifiers",file=sys.stderr)


    def counttranslations(self, reader):
        tcount = defaultdict( lambda: defaultdict(int) )
        for sentencepair in reader:
            for left, sourcefragment, right in sentencepair.inputfragments():
                targetfragment = sentencepair.reffragmentsdict()[sourcefragment.id]
                tcount[str(sourcefragment)][str(targetfragment)] += 1
        return tcount

    def countkeywords(self, reader, keywords, compute_bow_params, bow_absolute_threshold, bow_prob_threshold,bow_filter_threshold):
        print("Counting words for keyword extraction...", file=sys.stderr)
        wcount = defaultdict(int)
        wcount_total = 0
        tcount = defaultdict( lambda: defaultdict(int) )
        kwcount = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for sentencepair in reader:
            if int(sentencepair.id) % 1000 == 0:
                print(" Counting words for keyword extraction @" + str(sentencepair.id), file=sys.stderr)
            for left, sourcefragment, right in sentencepair.inputfragments():
                for word in sourcefragment:
                    wcount[word] += 1
                    wcount_total += 1
                for word in itertools.chain(left.split(), right.split()):
                    wcount[word] += 1
                    wcount_total += 1
                    targetfragment = sentencepair.reffragmentsdict()[sourcefragment.id]
                    tcount[str(sourcefragment)][str(targetfragment)] += 1
                    kwcount[str(sourcefragment)][str(targetfragment)][word] += 1


        return wcount, tcount,kwcount, wcount_total


    def probability_translation_given_keyword(self, source,target, keyword, kwcount, wcount):
        source = str(source)
        target = str(target)

        if not source in kwcount:
            print("focusword not seen", source, file=sys.stderr)
            return 0 #focus word has not been counted for

        if not target in kwcount[source]:
            print("target not seen:", target, file=sys.stderr)
            return 0 #sense has never been seen for this focus word

        if not keyword in wcount:
            print("keyword not seen:", keyword, file=sys.stderr)
            return 0 #keyword has never been seen

        Ns_kloc = 0.0
        if keyword in kwcount[source][target]:
            Ns_kloc = float(kwcount[source][target][keyword])

        Nkloc = 0
        for t in kwcount[source]:
            if keyword in kwcount[source][t]:
                Nkloc += kwcount[source][t][keyword]


        Nkcorp = wcount[keyword] #/ float(totalcount_sum)

        return (Ns_kloc / Nkloc) * (1/Nkcorp)


    def extract_keywords(self, sourcefragment, bow_absolute_threshold, bow_prob_threshold, bow_filter_threshold, kwcount, wcount):
        print("Extracting keywords for " + sourcefragment + "...", file=sys.stderr)

        sourcefragment = str(sourcefragment)

        if not sourcefragment in kwcount:
            print("WARNING: No count found!",file=sys.stderr)
            return [] #focus word has not been counted for


        bag = []
        #select all words that occur at least 3 times for a sense, and have a probability_sense_given_keyword >= 0.001
        for targetfragment in kwcount[sourcefragment]:
            for keyword, freq in kwcount[sourcefragment][targetfragment].items():
                if (wcount[keyword] >= bow_filter_threshold): #filter very rare words (occuring less than 20 times)
                    if freq>= bow_absolute_threshold:
                        p = self.probability_translation_given_keyword(sourcefragment, targetfragment, keyword, kwcount, wcount)
                        if p >= bow_prob_threshold:
                            bag.append( (keyword, targetfragment, freq, p) )

        if bag:
            print("\tFound " + str(len(bag)) + " keywords: ", file=sys.stderr)
            bag = sorted(bag)
            f = open(self.workdir + '/' + quote_plus(sourcefragment) + '.keywords','w',encoding='utf-8')
            for keyword, targetfragment, c, p in bag:
                f.write(keyword + '\t' + str(targetfragment) + '\t' + str(c) + '\t' + str(p) + '\n')
                print("\t\t", keyword, file=sys.stderr)
            f.close()

        else:
            print("\tNo keywords found", file=sys.stderr)
        return bag





    def build(self, reader, leftcontext, rightcontext, dokeywords, compute_bow_params, bow_absolute_threshold, bow_prob_threshold,bow_filter_threshold, timbloptions, limit=None):
        assert (isinstance(reader, Reader))


        if dokeywords:
            print("Counting keywords", file=sys.stderr)
            wcount, tcount, kwcount, wcount_total = self.countkeywords(reader, dokeywords, compute_bow_params, bow_absolute_threshold, bow_prob_threshold,bow_filter_threshold)
        else:
            print("Gathering initial occurrence count", file=sys.stderr)
            tcount = self.counttranslations(reader)
            wcount = {} #not needed
            kwcount = {} #not needed

        reader.reset()

        #make translation table of direct translation that have only one translation
        print("Writing direct translation table", file=sys.stderr)
        dttable = open(self.workdir + '/directtranslation.table','w',encoding='utf-8')
        for source in tcount:
            for target in tcount[source]:
                if len(tcount[str(source)]) == 1:
                    dttable.write(str(source) + "\t" + str(target) + "\t" + str(tcount[str(source)][str(target)]) + "\n")
            #gather keywords:
            if dokeywords:
                bag = self.extract_keywords(source, bow_absolute_threshold, bow_prob_threshold, bow_filter_threshold, kwcount, wcount)
                if bag:
                    self.keywords[source] = bag
        dttable.close()



        if not dokeywords and not leftcontext and not rightcontext:
            print("No classifiers needed, skipping...", file=sys.stderr)
            return

        index = open(self.workdir + '/index.table','w',encoding='utf-8')
        #now loop over corpus and build classifiers for those where disambiguation is needed
        for sentencepair in reader:
            targetfragments = sentencepair.reffragmentsdict()
            usedclassifier = False
            for left, inputfragment, right in sentencepair.inputfragments():
                if not str(inputfragment) in tcount:
                    print("WARNING: Inputfragment " + str(inputfragment) + " not found in count! Skipping!!", file=sys.stderr)
                    continue
                if limit and not str(inputfragment) in limit:
                    print("NOTICE: Inputfragment " + str(inputfragment) + " not found in testset, skipping...", file=sys.stderr)
                    continue
                left = tuple(left.split())
                right = tuple(right.split())
                if len(tcount[str(inputfragment)]) > 1:
                    usedclassifier = True
                    #extract local context
                    features = []
                    f_left = []
                    if leftcontext:
                        f_left = list(left[-leftcontext:])
                        if len(f_left) < leftcontext:
                            f_left = list(["<s>"] * (leftcontext - len(f_left))) + f_left
                    features += f_left

                    f_right = []
                    if rightcontext:
                        f_right = list(right[:rightcontext])
                        if len(f_right) < rightcontext:
                            f_right = f_right + list(["</s>"] * (rightcontext - len(f_right)))
                    features += f_right

                    targetfragment = targetfragments[inputfragment.id]

                    #extract global context
                    if dokeywords and str(inputfragment) in self.keywords:
                        bag = {}
                        for keyword, target, freq,p in self.keywords[str(inputfragment)]:
                            bag[keyword] = 0
                            if len(bag) == MAXKEYWORDS:
                                break

                        for word in itertools.chain(left, right):
                            if word in bag:
                                bag[keyword] = 1

                        #add to features
                        for keyword in sorted(bag.keys()):
                            features.append(keyword+"="+str(bag[keyword]))

                    if not str(inputfragment) in self.classifiers:
                        #Build classifier
                        cid = quote_plus(str(inputfragment))
                        cidfile = self.workdir + '/' +  cid
                        print("\tInitiating classifier for " + str(inputfragment) + " -- " + cidfile + '.train',file=sys.stderr )
                        self.classifiers[str(inputfragment)] = timbl.TimblClassifier(cidfile, timbloptions)
                        index.write(cid + "\t" + str(inputfragment) + "\t" + str(sum(tcount[str(inputfragment)].values())) + "\t")
                        for target, c in tcount[str(inputfragment)].items():
                            index.write(str(target) + "\t" + str(c))
                        index.write("\n")
                        configid = 'l' + str(leftcontext) + 'r' + str(rightcontext)
                        if dokeywords: configid += 'k'
                        f = open(self.classifiers[str(inputfragment)].fileprefix + '.conf', 'w',encoding='utf-8')
                        f.write("config=" + configid+"\n")
                        f.close()

                    if not features:
                        print("WARNING: No features extracted for " + str(inputfragment) + ", skipping classifier!", file=sys.stderr)
                    else:
                        print("\tAppending training instance: " +str(inputfragment) + " -> " + str(targetfragment),file=sys.stderr )
                        self.classifiers[str(inputfragment)].append( features, str(targetfragment) )

            if usedclassifier:
                print("Built classifier(s) for @" + str(sentencepair.id), file=sys.stderr)
            elif int(sentencepair.id) % 1000 == 0:
                print("Skipped @" + str(sentencepair.id), file=sys.stderr)
        index.close()
        for i, classifier in enumerate(self.classifiers):
            self.classifiers[classifier].flush()



    def gettimblskipopts(self, classifier, leftcontext,rightcontext,newleftcontext,newrightcontext, skipkeywords):
        skip = []
        for i in range(1,leftcontext+rightcontext+1):
            if i <= leftcontext:
                if i <= leftcontext - newleftcontext:
                    skip.append(i)
            elif i - rightcontext <= rightcontext:
                if i - leftcontext > newrightcontext:
                    skip.append(i)

        if not skip and not skipkeywords:
            return ""

        o =  "-mO:I" + ",".join([ str(i) for i in skip ])

        if skipkeywords and classifier in self.keywords:
            if skip: o += ","
            l = min(MAXKEYWORDS, len( set( ( x[0] for x in self.keywords[classifier] ) ) ) )
            if l == 1:
                o += str(leftcontext+rightcontext+1)
            else:
                skip = []
                for i in range(leftcontext+rightcontext+1, leftcontext+rightcontext+l + 1):
                    skip.append(i)
                o +=  ",".join([ str(i) for i in skip ])
        if o[-2:] == ":I":
            return ""
        return o

    def crossvalidate(self, classifier, folds, leftcontext, rightcontext, dokeywords,  newleftcontext, newrightcontext, newdokeywords, timbloptions):
        assert newleftcontext <= leftcontext
        assert newrightcontext <= rightcontext
        timblskipopts = self.gettimblskipopts(classifier, leftcontext, rightcontext, newleftcontext, newrightcontext, dokeywords and not newdokeywords )

        timbloptions = timbloptions.replace("-vdb","")
        timbloptions = timbloptions.replace("-G0","")


        #check that the number of lines exceeds the number of folds
        enoughlines = False
        trainfile = self.classifiers[classifier].fileprefix + ".train"
        fin = open(trainfile,'r',encoding='utf-8')
        linecount = 0
        for line in fin:
            linecount += 1
            if linecount > folds:
                enoughlines = True
                break
        if enoughlines:
            #ok, let's make us some folds!
            tmpid = self.classifiers[classifier].fileprefix
            fold = {}
            foldconfig = open(tmpid+ '.folds','w', encoding='utf-8') #will hold the fold-containing files, one per line.. as timbl likes it
            for i in range(0,folds):
                fold[i] = open(tmpid + '.fold' + str(i), 'w', encoding='utf-8')
                foldconfig.write(tmpid + '.fold' + str(i) + '\n')
            foldconfig.close()
            #make folds:
            fin.seek(0)
            for i, line in enumerate(fin):
                f = i % folds
                fold[f].write(line)
            for f in fold.values():
                f.close()
        fin.close()

        if not enoughlines:
            #Do simply leave one out
            c = timbl.TimblClassifier(self.classifiers[classifier].fileprefix, timbloptions + " " + timblskipopts)
            #c.train() not possible with LOO
            accuracy = c.leaveoneout()
        else:
            #Do cross validation
            c = timbl.TimblClassifier(self.classifiers[classifier].fileprefix, timbloptions + " " + timblskipopts)
            #c.train() #not possible with CV
            print("\tCross-validating using CV: " + tmpid + '.folds')
            accuracy = c.crossvalidate(tmpid + '.folds')

            #cleanup
            os.unlink(tmpid + '.folds')
            for i in range(0,folds):
                os.unlink(tmpid + '.fold' + str(i))
                if os.path.exists(tmpid + '.fold' + str(i) + '.cv'):
                    os.unlink(tmpid + '.fold' + str(i) + '.cv')
                else:
                    print("WARNING: No timbl output for fold " + str(i) + "!", file=sys.stderr)
            del c

        print("\tAccuracy=",accuracy, file=sys.stderr)
        return accuracy, timblskipopts




    def autoconf(self, folds,  leftcontext, rightcontext, dokeywords, timbloptions, limit=None):
        print("Auto-configuring " + str(len(self.classifiers)) + " classifiers, determining optimal feature configuration using leave-one-out", file=sys.stderr)
        l= len(self.classifiers)
        for i, classifier in enumerate(self.classifiers):
            if limit and not classifier in limit:
                print("NOTICE: Inputfragment " + classifier + " not found in testset, skipping...", file=sys.stderr)
                continue
            self.classifiers[classifier].flush()
            best = 0
            configid = 'l' + str(leftcontext) + 'r' + str(rightcontext)
            if dokeywords: configid += "k"
            bestconfig = (leftcontext,rightcontext,dokeywords,"")
            print("=================== #" + str(i+1) + "/" + str(l) + " - Autoconfiguring '" + classifier + "' (" + datetime.datetime.now().strftime("%H:%M:%S") + ") ===================", file=sys.stderr)
            for c in range(1,max(leftcontext,rightcontext)+1):
                print("- - - - - - - - - - - - Testing '" + classifier + "' with configuration l" + str(c) + "r" + str(c) + " - - - - - - - - - - -", file=sys.stderr)
                sys.stderr.flush()
                accuracy, timblskipopts = self.crossvalidate(classifier, folds, leftcontext, rightcontext, dokeywords,  c, c, False, timbloptions)
                if accuracy > best:
                    bestconfig = (c,c,False, timblskipopts)
                    best = accuracy
                if dokeywords:
                    print("- - - - - - - - - - - - Testing '" + classifier + "' with configuration l" + str(c) + "r" + str(c) + "k - - - - - - - - - - -", file=sys.stderr)
                    sys.stderr.flush()
                    accuracy, timblskipopts = self.crossvalidate(classifier, folds, leftcontext, rightcontext, dokeywords, c, c, True, timbloptions)
                    if accuracy > best:
                        bestconfig = (c,c,False, timblskipopts)
                        best = accuracy
            if best == 0:
                bestconfigid = 'l1r1'
            else:
                bestconfigid = 'l' + str(bestconfig[0]) + 'r' + str(bestconfig[1])
                if bestconfig[2]: bestconfigid += 'k'

            f = open(self.classifiers[classifier].fileprefix + '.conf', 'w',encoding='utf-8')
            f.write("config=" + configid+"\n")
            f.write("bestconfig=" + bestconfigid+"\n")
            f.write("timblopts=" + bestconfig[3] + "\n")
            f.write("accuracy=" + str(best) + "\n")
            f.close()
            print("\tBest configuration for '" + classifier + "' is " + configid + " with accuracy " + str(best), file=sys.stderr)


    def readconf(self, classifier):
        configid = ""
        bestconfigid = ""
        timblopts = ""
        accuracy = 0.0
        f = open(self.classifiers[classifier].fileprefix + '.conf', 'r',encoding='utf-8')
        for line in f:
            line = line.strip()
            if line[0:7] == 'config=':
                configid = line[7:]
            elif line[0:11] == 'bestconfig=':
                bestconfigid = line[11:]
            elif line[0:10] == 'timblopts=':
                timblopts = line[10:]
            elif line[0:9] == 'accuracy=':
                accuracy = float(line[9:])
            elif line and line[0] != '#':
                raise ValueError("readconf(): Unable to parse: " + line)
        f.close()
        return configid, bestconfigid, timblopts, accuracy

    def train(self, leftcontext, rightcontext, dokeywords, limit=None):
        print("Training " + str(len(self.classifiers)) + " classifiers", file=sys.stderr)
        for classifier in self.classifiers:
            if limit and not classifier in limit:
                print("NOTICE: Inputfragment " + classifier + " not found in testset, skipping...", file=sys.stderr)
                continue
            self.classifiers[classifier].flush()
            if os.path.exists(self.classifiers[classifier].fileprefix + '.conf'):
                configid, bestconfigid, timblopts, accuracy = self.readconf(classifier)
                if timblopts: self.classifiers[classifier].timbloptions += ' ' + timblopts
                if bestconfigid: print("\tSelected configuration " + bestconfigid + " for '" + classifier + "' (" + configid + ")", file=sys.stderr)
            else:
                f = open(self.classifiers[classifier].fileprefix + '.conf', 'w',encoding='utf-8')
                f.write("config=l" + str(leftcontext) + 'r' + str(rightcontext))
                if dokeywords:
                    f.write('k')
                f.write("\n")
                f.close()

            if os.path.exists(self.classifiers[classifier].fileprefix + '.train'):
                print("\tTraining '" + classifier + "'", file=sys.stderr)
                self.classifiers[classifier].train()
                self.classifiers[classifier].save()



    def classify(self, inputfragment, left, right, sentencepair, generalleftcontext, generalrightcontext, generaldokeywords, timbloptions, lm=None,tweight=1,lmweight=1):
        #translation by classifier
        classifier = self.classifiers[str(inputfragment)]


        if not (classifier.selectedleftcontext is None):
            leftcontext = classifier.selectedleftcontext
        else:
            leftcontext = generalleftcontext

        if not (classifier.selectedrightcontext is None):
            rightcontext = classifier.selectedrightcontext
        else:
            rightcontext = generalrightcontext

        if not (classifier.selectedkeywords is None):
            dokeywords = classifier.selectedkeywords
        else:
            dokeywords = generaldokeywords

        features = []

        if leftcontext or classifier.leftcontext:
            f_left = list(left[-leftcontext:])
            if len(f_left) < leftcontext:
                f_left = list(["<s>"] * (leftcontext - len(f_left))) + f_left
            if not (classifier.leftcontext is None):
                if classifier.leftcontext < leftcontext:
                    f_left = f_left[-classifier.leftcontext:]
                elif leftcontext < classifier.leftcontext:
                    f_left = list(["<DUMMY-IGNORED>"] * (classifier.leftcontext - leftcontext)) + f_left
            features += f_left

        features.append(str(inputfragment))

        if rightcontext or classifier.rightcontext:
            f_right = list(right[:rightcontext])
            if len(f_right) < rightcontext:
                f_right = f_right + list(["</s>"] * (rightcontext - len(f_right)))
            if not (classifier.rightcontext is None):
                if classifier.rightcontext < rightcontext:
                    f_right = f_right[:classifier.rightcontext]
                elif rightcontext < classifier.rightcontext:
                    f_right = f_right + list(["<DUMMY-IGNORED>"] * (classifier.rightcontext - rightcontext))
            features += f_right


        #extract global context
        keywordsfound = 0
        if str(inputfragment) in self.keywords:
            if dokeywords:
                bag = {}
                for keyword, target, freq,p in self.keywords[str(inputfragment)]:
                    bag[keyword] = 0
                    if len(bag) == MAXKEYWORDS:
                        break

                #print("Bag", repr(bag), file=sys.stderr)
                for word in itertools.chain(left, right):
                    #print(repr(word),file=sys.stderr)
                    if word in bag:
                        if bag[word] == 0:
                            keywordsfound += 1
                        bag[word] = 1

                #add to features
                for keyword in sorted(bag.keys()):
                    features.append(keyword+"="+str(bag[keyword]))
            elif classifier.keywords: #classifier was trained with keywords, need dummies
                for i, keyword in enumerate( set( ( x[0] for x in self.keywords[str(inputfragment)]) ) ):
                    if i == MAXKEYWORDS: break
                    features.append("<IGNOREDKEYWORD"+str(i+1)+">")

        #pass to classifier
        if keywordsfound > 0:
            print("\tClassifying '" + str(inputfragment) + "' (" + str(keywordsfound) + " keywords found)...", file=sys.stderr)
        else:
            print("\tClassifying '" + str(inputfragment) + "' ...", file=sys.stderr)
        if classifier.leftcontext != leftcontext or classifier.rightcontext != rightcontext:
            print("\t\tClassifier configuration: l:",classifier.leftcontext,"r:",classifier.rightcontext," || Desired configuration: l:",leftcontext,"r:",rightcontext, " || Timbloptions: ", classifier.timbloptions, file=sys.stderr)
        print("\tFeature vector: " + " ||| ".join(features),file=sys.stderr)
        classlabel, distribution, distance =  classifier.classify(features)
        classlabel = classlabel.replace(r'\_',' ')
        if lm and len(distribution) > 1:
            print("\tClassifier translation prior to LM: " + str(inputfragment) + " -> [ DISTRIBUTION:" + str(repr(distribution))+" ]", file=sys.stderr)
            candidatesentences = []
            bestlmscore = -999999999
            besttscore = -999999999
            for targetpattern, score in distribution.items():
                assert score >= 0 and score <= 1
                tscore = math.log(score) #base-e log (LM is converted to base-e upon load)
                translation = tuple(targetpattern.split())
                outputfragment = Fragment(translation, inputfragment.id, score)
                candidatesentence = sentencepair.replacefragment(inputfragment, outputfragment, sentencepair.output)
                lminput = " ".join(sentencepair._str(candidatesentence)).split(" ") #joining and splitting deliberately to ensure each word is one item
                lmscore = lm.score(lminput)
                assert lmscore <= 0
                if lmscore > bestlmscore:
                    bestlmscore = lmscore
                if tscore > besttscore:
                    besttscore = tscore
                candidatesentences.append( ( candidatesentence, outputfragment, tscore, lmscore ) )

            #get the strongest sentence
            maxscore = -9999999999
            for candidatesentence, targetpattern, tscore, lmscore in candidatesentences:
                tscore = tweight * (tscore-besttscore)
                lmscore = lmweight * (lmscore-bestlmscore)
                score = tscore + lmscore
                print("\t LM candidate " + str(inputfragment) + " -> " + str(targetpattern) + "   score=tscore+lmscore=" + str(tscore) + "+" + str(lmscore) + "=" + str(score), file=sys.stderr)
                if score > maxscore:
                    maxscore = score
                    outputfragment = targetpattern  #Fragment(targetpattern, inputfragment.id)
                    outputfragment.confidence = score
            for candidatesentence, targetpattern, tscore, lmscore in candidatesentences:
                if targetpattern != outputfragment:
                    outputfragment.alternatives.append( Alternative( tuple(str(targetpattern).split()), tweight* (tscore-besttscore) + lmweight * (lmscore-bestlmscore) )  )
            print("\tClassifier translation after LM: " + str(inputfragment) + " -> " + str(outputfragment) + " score= " + str(score), file=sys.stderr)

        else:
            outputfragment = Fragment(tuple(classlabel.split()), inputfragment.id, max(distribution.values()))
            for targetpattern, score in distribution.items():
                tscore = math.log(score) #convert to base-e log (LM is converted to base-e upon load)
                if targetpattern != classlabel:
                    outputfragment.alternatives.append( Alternative( tuple(targetpattern.split()), score) )
            print("\tClassifier translation " + str(inputfragment) + " -> " + str(outputfragment) + "\t[ DISTRIBUTION:" + str(repr(distribution))+" ]", file=sys.stderr)

        return outputfragment

    def processsentence(self, sentencepair, ttable, sourceclassencoder, targetclassdecoder, generalleftcontext, generalrightcontext, generaldokeywords, timbloptions, lm=None,tweight=1,lmweight=1, dofragmentdecode=True):
        print("Processing sentence " + str(sentencepair.id),file=sys.stderr)
        sentencepair.ref = None
        sentencepair.output = copy(sentencepair.input)

        for left, inputfragment, right in sentencepair.inputfragments():
            inputfragment_s = str(inputfragment)
            print("\tFragment: ", inputfragment_s, file=sys.stderr)
            try:
                inputfragment_p = sourceclassencoder.buildpattern(inputfragment_s)
            except IOError:
                print("\tNOTICE: One or more words in '" + inputfragment_s + "' were not seen during training",file=sys.stderr)
                inputfragment_p = None
            left = tuple(left.split())
            right = tuple(right.split())
            if inputfragment_s in self.classifiers:
                outputfragment =  self.classify(inputfragment, left, right, sentencepair, generalleftcontext, generalrightcontext, generaldokeywords, timbloptions, lm,tweight,lmweight)
            elif ttable and inputfragment_p and inputfragment_p in ttable:
                outputfragment = None
                for targetpattern, scores in sorted(ttable[inputfragment_p].items(),key=lambda x: -1* x[1][2]):
                    targetpattern_s = targetpattern.tostring(targetclassdecoder)
                    outputfragment = Fragment(tuple( targetpattern_s.split(' ') ), inputfragment.id )
                    print("\tFallback translation from phrasetable" + str(inputfragment) + " -> " + str(outputfragment), file=sys.stderr)
                    break
                if outputfragment is None:
                    raise Exception("No outputfragment found in phrasetable!!! Shouldn't happen")
            #elif dofragmentdecode and len(inputfragment) > 1:
                #print("\tFragment not directly translatable: " + str(inputfragment), file=sys.stderr)
                #solutions = []
                #for fragmentation in self.decodefragments(inputfragment, dttable):
                #    translatedfragmentation = []
                #    for fragment in fragmentation:
                #        if fragment in dttable:
                #            translatedfragmentation.append(dttable[fragment])
                #        elif fragment in self.classifiers:
                #            translatedfragmentation.append( self.classify( left + tuple(translatedfragmentation), tuple(['{UNKNOWN}'] * 10), sentencepair, dttable, generalleftcontext, generalrightcontext, generaldokeywords, timbloptions, lm,tweight,lmweight) )
                #    solutions.append(translatedfragmentation)
                pass
            else:
                #no translation found
                outputfragment = Fragment(None, inputfragment.id)
                print("\tNo translation for " + inputfragment_s, file=sys.stderr)
            sentencepair.output = sentencepair.replacefragment(inputfragment, outputfragment, sentencepair.output)
        return sentencepair

    def loaddttable(self):
        return loaddttable(self.workdir + '/directtranslation.table')

    def test(self, data, outputfile, ttable, sourceclassencoder, targetclassdecoder, leftcontext, rightcontext, dokeywords, timbloptions, lm=None,tweight=1,lmweight=1, dofragmentdecode=True):
        writer = Writer(outputfile)
        for sentencepair in data:
            sentencepair = self.processsentence(sentencepair, ttable, sourceclassencoder, targetclassdecoder, leftcontext, rightcontext, dokeywords, timbloptions, lm, tweight, lmweight, dofragmentdecode)
            writer.write(sentencepair)
        writer.close()


    def decodefragments(self, inputfragment, dttable):
        assert isinstance(inputfragment, tuple)
        results = 0
        for i in reversed(range(1,len(inputfragment))):
            subfragment = " ".join(inputfragment[0:i])
            if subfragment in self.classifiers or subfragment in dttable:
                fragmentation = [subfragment]
                if i == len(inputfragment) - 1:
                    yield fragmentation
                    results += 1
                else:
                    #recursion
                    for fragmentation in self.decodefragments(inputfragment[i+1:],dttable):
                        yield subfragment + fragmentation


#class MosesModel:
#    def __init__(self, workdir, ttable):
#        self.workdir = workdir
#        self.ttable = ttable




def loaddttable(filename):
    dttable = {}
    f = open(filename)
    for line in f:
        if line:
            fields = line.split('\t')
            dttable[fields[0]] = fields[1]
    f.close()
    return dttable

def getlimit(testset):
    limit = set()
    for sentencepair in testset:
        for left, sourcefragment, right in sentencepair.inputfragments():
            limit.add( str(sourcefragment) )
    return limit



def main():
    parser = argparse.ArgumentParser(description="Colibrita - Translation Assistance", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--trainfromset',type=str,help="Training mode using pregenerated training set (use with -f)", action='store',default="")
    parser.add_argument('--train',help="Build classifiers and train from scratch based on a parallel corpus", action='store_true')
    parser.add_argument('--source', type=str,help="Source language corpus for training (plaintext)", action='store',required=False)
    parser.add_argument('--target', type=str,help="Target language corpus for training (plaintext)", action='store',required=False)
    parser.add_argument('-M','--phrasetable', type=str,help="Moses phrasetable to use for training (--train)", action='store',default="")
    parser.add_argument('--trainfortest',type=str, help="Do only limited training that covers a particular test set (speeds up training and reduces memory considerably!), use with --train or --trainfromset", action='store',default="")
    parser.add_argument('--test',type=str,help="Test mode (against a specific test set)", action='store',default="")
    parser.add_argument('--baseline', type=str,help="Baseline test on specified test set", action='store',default="")
    parser.add_argument('--run',help="Run mode (reads input from stdin)", action='store_true')
    parser.add_argument('--server', help="Server mode (RESTFUL HTTP Server)", action='store_true')
    #parser.add_argument('--igen',dest='settype',help="Instance generation from a training set (-f) without actual training", action='store_const',const='igen')
    parser.add_argument('--debug','-d', help="Debug", action='store_true', default=False)
    parser.add_argument('-a','--autoconf', help="Automatically determine best feature configuration per expert (cross-validated), values for -l and -r are considered maxima, set -k to consider keywords, needs to be specified both at training time and at test time!", action='store_true',default=False)
    parser.add_argument('-l','--leftcontext',type=int, help="Left local context size", action='store',default=0)
    parser.add_argument('-r','--rightcontext',type=int,help="Right local context size", action='store',default=0)

    parser.add_argument('--maxlength',type=int,help="Maximum length of phrases", action='store',default=10)
    parser.add_argument('-k','--keywords',help="Add global keywords in context", action='store_true',default=False)
    parser.add_argument('-F','--decodefragments',help="Attempt to decode long unknown fragments by breaking it up into smaller parts (not implemented yet!)", action='store_true',default=False)
    parser.add_argument("--kt",dest="bow_absolute_threshold", help="Keyword needs to occur at least this many times in the context (absolute number)", type=int, action='store',default=3)
    parser.add_argument("--kp",dest="bow_prob_threshold", help="minimal P(translation|keyword)", type=int, action='store',default=0.001)
    parser.add_argument("--kg",dest="bow_filter_threshold", help="Keyword needs to occur at least this many times globally in the entire corpus (absolute number)", type=int, action='store',default=20)
    parser.add_argument("--ka",dest="compute_bow_params", help="Attempt to automatically compute --kt,--kp and --kg parameters", action='store_false',default=True)
    #parser.add_argument('-O', dest='timbloptions', help="Timbl Classifier options", type=str,action='store',default="-k 1")
    parser.add_argument('--Tk', dest='timbl_k', help="Timbl k", type=int,action='store',default=1)
    parser.add_argument('--Tclones', dest='timbl_clones', help="Timbl clones (number of CPUs to use for parallel processing)", type=int,action='store',default=1)
    parser.add_argument('-o','--output',type=str,help="Output prefix", required = True)
    #parser.add_argument('--moses', help="Use Moses as Translation Model, no classifiers will be used, use with -T", action='store_true',default=False)
    parser.add_argument('--lm',type=str, help="Use language model in testing (file in ARPA format, as produced by for instance SRILM)", action='store',default="")
    parser.add_argument('--lmweight',type=float, help="Language model weight (when --lm is used)", action='store',default=1)
    parser.add_argument('--tmweight',type=float, help="Translation model weight (when --lm is used)", action='store',default=1)
    parser.add_argument('--port',type=int, help="Server port (use with --server)", action='store',default=7893)
    parser.add_argument('--folds',type=int, help="Number of folds to use in for cross-validatio (used with -a)", action='store',default=10)
    parser.add_argument('-T','--ttable', type=str,help="Phrase translation table (file) to use, must be a Colibri alignment model (use colibri-mosesphrasetable2alignmodel). Will be tried as a fallback when no classifiers are made, also required when testing with --lm and without classifier training, and when using --trainfromscratch", action='store',default="")

    #setgen options
    parser.add_argument('-p', dest='joinedprobabilitythreshold', help="Used with --trainfromscratch: Joined probabiity threshold for inclusion of fragments from phrase translation-table: min(P(s|t) * P(t|s))", type=float,action='store',default=0)
    parser.add_argument('-D', dest='divergencefrombestthreshold', help="Used with --trainfromscratch: Maximum divergence from best translation option. If set to 0.8, the only alternatives considered are those that have a joined probability of equal or above 0.8 of the best translation option", type=float,action='store',default=0)

    parser.add_argument('--mosesdir',type=str, help="Path to moses (for --trainfromscratch)",action='store',default="")
    parser.add_argument('--bindir',type=str, help="Path to external bin dir (path where moses bins are installed, for --trainfromscratch)",action='store',default="/usr/local/bin")

    args = parser.parse_args()


    if not args.train and not args.test and not args.trainfromset and not args.trainfortest:
        print("Specify either --train, --test, --trainfromset, --trainfortst")
        sys.exit(2)


    timbloptions = "-vdb -G0 -k " + str(args.timbl_k) #if I add -a 0 explicitly (default anyway)   leave_one_out and crossvalidate won't work!!!
    if args.timbl_clones > 1:
        timbloptions += " --clones=" + str(args.timbl_clones)




    if args.trainfromset:


        ####################### TRAIN FROM XML SET (old method) #################################3
        #TODO!! ADAPT TO NEW COLIBRI-MT STYLE!

        if args.lm:
            print("WARNING: Language model specified during training, will be ignored", file=sys.stderr)

        print("Parameters: ", repr(args), file=sys.stderr)

        experts = ClassifierExperts(args.output)
        if args.trainfortest:
            testset = Reader(args.trainfortest)
            limit = getlimit(testset)
        else:
            limit = None

        data = Reader(args.trainfromset)
        if not os.path.isdir(args.output):
            os.mkdir(args.output)
        if not os.path.exists(args.output + '/phrasetable.colibri.alignmodel-keys'):
            print("Creating alignment model from Moses phrasetable, unconstrained",file=sys.stderr)
            r = os.system("colibri-mosesphrasetable2alignmodel -i " + args.phrasetable + " -o " + args.output + "/colibri.alignmodel -j " + str(args.joinedprobabilitythreshold) + " -d " + str(args.divergencefrombestthreshold) + " -S " + args.sourceclassfile + " -T " + args.targetclassfile)
            if r != 0:
                print("Failed",file=sys.stderr)
                sys.exit(2)

            print("Building classifiers", file=sys.stderr)
            experts.build(data, args.leftcontext, args.rightcontext, args.keywords, args.compute_bow_params, args.bow_absolute_threshold, args.bow_prob_threshold, args.bow_filter_threshold, timbloptions, limit)
        else:
            print("Classifiers already built", file=sys.stderr)
            experts.load(timbloptions, args.leftcontext, args.rightcontext, args.keywords, limit, args.autoconf)
        if args.autoconf:
            experts.autoconf(args.folds, args.leftcontext, args.rightcontext, args.keywords, timbloptions, limit)
        experts.train(args.leftcontext, args.rightcontext, args.keywords, limit)



    elif args.train or args.trainfortest:
        ####################### TRAIN FROM SCRATCH #################################3
        if not os.path.isdir(args.output):
            os.mkdir(args.output)
        if not args.source:
            print("--train requires parameter --source",file=sys.stderr)
            sys.exit(2)
        if not args.target:
            print("--train requires parameter --target",file=sys.stderr)
            sys.exit(2)
        if not args.phrasetable or not os.path.exists(args.phrasetable):
            print("--train required a Moses phrase-translation table (-M). Not Found",file=sys.stderr)
            sys.exit(2)

        sourceclassfile  = args.source.replace('.txt','') + '.colibri.cls'
        sourcecorpusfile  = args.source.replace('.txt','') + '.colibri.dat'
        sourcemodelfile  = args.source.replace('.txt','') + '.colibri.indexedpatternmodel'
        targetclassfile = args.target.replace('.txt','') + '.colibri.cls'
        targetcorpusfile  = args.target.replace('.txt','') + '.colibri.dat'
        targetmodelfile  = args.target.replace('.txt','') + '.colibri.indexedpatternmodel'

        conf = {'sourceclassfile':sourceclassfile, 'targetclassfile': targetclassfile,'sourcecorpusfile':sourcecorpusfile,'targetcorpusfile': targetcorpusfile}
        pickle.dump(conf, open(args.output+'/colibrita.conf','wb'))

        if not os.path.exists(sourcecorpusfile) or not os.path.exists(sourceclassfile):
            print("1.1) Encoding source corpus",file=sys.stderr)
            r = os.system("colibri-classencode " + args.source)
            if r != 0:
                print("Failed",file=sys.stderr)
                sys.exit(2)



        if not os.path.exists(targetcorpusfile) or not os.path.exists(targetclassfile):
            print("1.2) Encoding target corpus",file=sys.stderr)
            r = os.system("colibri-classencode " + args.target)
            if r != 0:
                print("Failed",file=sys.stderr)
                sys.exit(2)





        if args.trainfortest:
            if not os.path.exists(args.output + "/testfragments.colibri.unindexedpatternmodel"):
                print("2.1) Reading test fragments",file=sys.stderr)
                with open(args.output + "/testfragments.txt",'w',encoding='utf-8') as f:
                    for sentencepair in Reader(args.trainfortest):
                        for fragment in sentencepair.fragments(sentencepair.input, True).values():
                            fragment = " ".join(fragment.value)
                            f.write(fragment  +"\n")

                print("2.2) Encoding test fragments",file=sys.stderr)
                r = os.system("colibri-classencode -e -c " + sourceclassfile + " -d " + args.output + "/ " + args.output + "/testfragments.txt")
                if r != 0:
                    print("Failed",file=sys.stderr)
                    sys.exit(2)
                os.rename(args.output + "/testfragments.colibri.cls", args.output + "/source.colibri.cls")
                sourceclassfile =  args.output + "/source.colibri.cls"


                #train an unindexed patternmodel model on testdata, to be used as constraint for the alignment model
                print("2.3) Building patternmodel on fragments in testdata",file=sys.stderr)
                r = os.system("colibri-patternmodeller -u -t 1 -l " + str(args.maxlength) + " -f " + args.output+"/testfragments.colibri.dat -o " + args.output+"/testfragments.colibri.unindexedpatternmodel")
                if r != 0:
                    print("Failed",file=sys.stderr)
                    sys.exit(2)

                os.unlink(args.output+"/testfragments.txt")

            if not os.path.exists(args.output+"/colibri.alignmodel"):
                #Convert moses phrasetable to alignment model, constrained by testset
                #no normalisation!
                cmd = "colibri-mosesphrasetable2alignmodel -N -i " + args.phrasetable + " -m " + args.output+"/testfragments.colibri.unindexedpatternmodel -o " + args.output + "/colibri.alignmodel -j " + str(args.joinedprobabilitythreshold) + " -d " + str(args.divergencefrombestthreshold) + " -S " + sourceclassfile + " -T " + targetclassfile
                print("2.4) Creating alignment model from Moses phrasetable, constrained by testset: " + cmd,file=sys.stderr)
                r = os.system(cmd)
                if r != 0:
                    print("Failed",file=sys.stderr)
                    sys.exit(2)
        else:
            #not constrained by testset

            if not os.path.exists(args.output+"/colibri.alignmodel"):
                #Convert moses phrasetable to alignment model, unconstrained by testset. No normalisation
                cmd = "colibri-mosesphrasetable2alignmodel -N -i " + args.phrasetable + " -o " + args.output + "/colibri.alignmodel -j " + str(args.joinedprobabilitythreshold) + " -d " + str(args.divergencefrombestthreshold) + " -S " + sourceclassfile + " -T " + targetclassfile
                print("2) Creating alignment model from Moses phrasetable, unconstrained by testset:" + cmd,file=sys.stderr)
                r = os.system(cmd)
                if r != 0:
                    print("Failed",file=sys.stderr)
                    sys.exit(2)



        #Extractfeatures needs indexed corpora to find context,  this may become huge so we constrain it as much as possible.
        #the alignment model serves as source-side constraint, for target-side we need to build another intermediate model from the alignment model:

        print("3.1) Loading alignment model",file=sys.stderr)
        alignmodel = AlignmentModel(args.output + "/colibri.alignmodel")
        print("\t" + str(len(alignmodel)) + " source-side patterns loaded",file=sys.stderr)

        print("3.1) Extracting target constraint model",file=sys.stderr)
        targetconstraintmodel = PatternSetModel()
        targetpatterncount = 0
        for pattern in alignmodel.targetpatterns():
            targetpatterncount += 1
            targetconstraintmodel.add(pattern)
        targetconstraintmodel.write(args.output + "/targetconstraints.colibri.patternsetmodel")
        print("\t" + str(targetpatterncount) + " target-side patterns found",file=sys.stderr)


        del alignmodel
        del targetconstraintmodel


        if not os.path.exists(sourcemodelfile):
            cmd = "colibri-patternmodeller -t 2 -l " + str(args.maxlength) + " -f " + sourcecorpusfile + " -o " + sourcemodelfile + " -j " + args.output + "/colibri.alignmodel"
            print("3.2) Building indexed pattern model on source corpus (constrained by phrasetable): ", cmd,file=sys.stderr)
            #threshold 2 is okay because extracting context features and building classifiers for patterns occuring only once is useless anyway
            r = os.system(cmd)
            if r != 0:
                print("Failed",file=sys.stderr)
                sys.exit(2)

        if not os.path.exists(targetmodelfile):
            print("3.3) Building indexed pattern model on target corpus (constrained by phrasetable): ", cmd, file=sys.stderr)
            #threshold 1 is needed here
            cmd = "colibri-patternmodeller -t 1 -l " + str(args.maxlength) + " -f " + targetcorpusfile + " -o " + targetmodelfile + " -j " + args.output + "/targetconstraints.colibri.patternsetmodel"
            r = os.system(cmd)
            if r != 0:
                print("Failed",file=sys.stderr)
                sys.exit(2)


        if not os.path.exists(args.output+'/classifier.conf'):
            cmd = "colibri-extractfeatures --crosslingual -C -X -i " + args.output + "/colibri.alignmodel -f " + targetcorpusfile + " -l " + str(args.leftcontext) + " -r " + str(args.rightcontext) + " -o " + args.output + " -s " + sourcemodelfile + " -t " + targetmodelfile + " -S " + sourceclassfile + " -T " + targetclassfile + " -c " + targetclassfile #TODO: do sourcemodel and targetmodel work with --crosslingual??
            print("4) Extracting features and building classifiers: " + cmd,file=sys.stderr)
            r = os.system(cmd)
            if r != 0:
                print("Failed",file=sys.stderr)
                sys.exit(2)
        else:
            print("Classifiers already generated",file=sys.stderr)


        if not os.path.exists(args.output + '/trained'):
            print("5) Loading and training classifiers",file=sys.stderr)
            limit = None
            experts = ClassifierExperts(args.output)
            experts.load(timbloptions, args.leftcontext, args.rightcontext, args.keywords, limit, args.autoconf)
            if args.autoconf:
                experts.autoconf(args.folds, args.leftcontext, args.rightcontext, args.keywords, timbloptions, limit)
            experts.train(args.leftcontext, args.rightcontext, args.keywords, limit)

            f = open(args.output + "/trained",'w')
            f.close()
        else:
            print("Classifiers already trained", file=sys.stderr)

            print("Training stage done", file=sys.stderr)



    if args.test:


        print("Parameters: ", repr(args), file=sys.stderr)


        if args.lm:
            print("Loading Language model", file=sys.stderr)
            lm = ARPALanguageModel(args.lm)
        else:
            lm = None

        if (args.leftcontext or args.rightcontext or args.keywords): # and not args.moses:
            if not os.path.isdir(args.output):
                print("Output directory " + args.output + " does not exist, did you forget to train the system first?", file=sys.stderr)
                sys.exit(2)
            if not os.path.exists(args.output + "/colibri.alignmodel"):
                print("Alignment model in output directory " + args.output + " does not exist, did you forget to train the system first?", file=sys.stderr)
                sys.exit(2)
            if not os.path.exists(args.output+'/colibrita.conf'):
                print("No colibrita.conf found in specified directory (-o). Has the system been trained?", file=sys.stderr)
                sys.exit(2)

            print("Loading configuration", file=sys.stderr)
            conf = pickle.load(open(args.output + '/colibrita.conf','rb'))
            sourceclassfile = conf['sourceclassfile']
            targetclassfile = conf['targetclassfile']

            print("Loading source class encoder", file=sys.stderr)
            sourceclassencoder = ClassEncoder(sourceclassfile)
            print("Loading target class decoder", file=sys.stderr)
            targetclassdecoder = ClassDecoder(targetclassfile)


            experts = ClassifierExperts(args.output)
            print("Loading classifiers",file=sys.stderr)
            experts.load(timbloptions, args.leftcontext, args.rightcontext, args.keywords, None, args.autoconf)

            print("Loading translation table (colibri alignment model)",file=sys.stderr)
            ttable = AlignmentModel(args.output + "/colibri.alignmodel");


            print("Running...",file=sys.stderr)
            data = Reader(args.test)
            experts.test(data, args.output + '.output.xml', ttable, sourceclassencoder,targetclassdecoder, args.leftcontext, args.rightcontext, args.keywords, timbloptions , lm,  args.tmweight, args.lmweight, args.decodefragments)

        elif args.baseline:
            print("Loading translation table",file=sys.stderr)
            ttable = AlignmentModel(args.output + "/colibri.alignmodel");
            #ttable = PhraseTable(args.ttable,False, False, "|||", 3, 0,None, None)

            data = Reader(args.baselene)
            print("Making baseline",file=sys.stderr)
            if args.lm:
                print("(with LM)",file=sys.stderr)
                makebaseline(ttable, args.output + '.output.xml', data, lm, args.tmweight, args.lmweight)
            elif args.baseline:
                makebaseline(ttable, args.output + '.output.xml', data)


        else:
            print("Don't know what to do! Specify some classifier options or -T with --lm or --baseline", file=sys.stderr)
    #elif args.settype == 'run' or args.settype == 'server':

    #    if args.settype == 'server':
    #        try:
    #            ColibritaServer
    #        except:
    #            print("Server not available, twisted not loaded...", file=sys.stderr)
    #            sys.exit(2)

    #    print("Parameters: ", repr(args), file=sys.stderr)
    #    if args.lm:
    #        print("Loading Language model", file=sys.stderr)
    #        lm = ARPALanguageModel(args.lm)
    #    else:
    #        lm = None

    #    if args.autoconf:
    #        print("Warning: Autoconf specified at testing time, has no effect. Has to be specified at training time", file=sys.stderr)

    #    experts = None
    #    dttable = {}
    #    ttable = None
    #    if args.leftcontext or args.rightcontext or args.keywords:
    #        if not os.path.exists(args.output + '/directtranslation.table'):
    #            print("Direct translation table does not exist, did you forget to train the system first?", file=sys.stderr)
    #            sys.exit(2)
    #        else:
    #            print("Loading direct translation table", file=sys.stderr)
    #            dttable = loaddttable(args.output + '/directtranslation.table')
    #        experts = ClassifierExperts(args.output)
    #        print("Loading classifiers",file=sys.stderr)
    #        experts.load(timbloptions, args.leftcontext, args.rightcontext, args.keywords)
    #    elif args.ttable:
    #        print("Loading translation table",file=sys.stderr)
    #        ttable = AlignmentModel()
    #        ttable.load(args.ttable)
            #ttable = PhraseTable(args.ttable,False, False, "|||", 3, 0,None, None)

        #if args.settype == 'run':
        #    print("Reading from standard input, enclose words/phrases in fallback language in asteriskes (*), type q<enter> to quit",file=sys.stderr)
        #    for line in sys.stdin:
        #        line = line.strip()
        #        if line == "q":
        #            break
        #        else:
        #            sentencepair = plaintext2sentencepair(line)
        #            if experts:
        #                sentencepair = experts.processsentence(sentencepair, ttable, args.leftcontext, args.rightcontext, args.keywords, timbloptions, lm, ttable, args.tmweight, args.lmweight, args.decodefragments)
        #            elif args.ttable:
        #                pass #TODO
        #            print(str(lxml.etree.tostring(sentencepair.xml(), encoding='utf-8',xml_declaration=False, pretty_print=True),'utf-8'), file=sys.stderr)
        #            print(sentencepair.outputstr())
        #elif args.settype == 'server':
        #    print("Starting Colibrita server on port " + str(args.port),file=sys.stderr)
        #    ColibritaServer(args.port, experts, dttable, ttable, lm, args, timbloptions)

    print("All done.", file=sys.stderr)


if __name__ == '__main__':
    main()
    sys.exit(0)

