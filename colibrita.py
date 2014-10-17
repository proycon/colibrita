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
import subprocess
import time
import socket
import lxml.etree
import signal
from collections import defaultdict
from urllib.parse import quote_plus, unquote_plus
from copy import copy

from colibrita.format import Writer, Reader, Fragment, Alternative
from colibrita.common import plaintext2sentencepair
from colibricore import ClassEncoder, ClassDecoder, PatternSetModel
from colibrimt.alignmentmodel import AlignmentModel
from pynlpl.lm.lm import ARPALanguageModel

import xmlrpc.client


try:
    from twisted.web import server, resource
    from twisted.internet import reactor

    class ColibritaProcessorResource(resource.Resource):
        isLeaf = True
        numberRequests = 0

        def __init__(self, experts, ttable,sourceclassencoder, targetclassdecoder, lm, args, timbloptions, mosesclient):
            self.experts = experts
            self.ttable = ttable
            self.lm = lm
            self.args = args
            self.timbloptions = timbloptions
            self.mosesclient = mosesclient
            self.sourceclassencoder = sourceclassencoder
            self.targetclassdecoder = targetclassdecoder

        def render_POST(self, request):
            return self.process(request)

        def render_GET(self, request):
            return self.process(request)

        def process(self, request):
            self.numberRequests += 1
            request.setHeader(b"Access-Control-Allow-Origin", b"*")
            if b'input' in request.args:
                request.setHeader(b"content-type", b"application/xml")
                print("Server Input: ", request.args, file=sys.stderr)
                print("Server input: ", request.args[b'input'][0], file=sys.stderr)
                line = str(request.args[b'input'][0],'utf-8')
                sentencepair = plaintext2sentencepair(line)
                if self.args.moses or self.args.mosesY or self.args.allornothing:
                    #classifier score
                    print("(Moses (-Z/-Y) after classifiers, passing full sentence)",file=sys.stderr)
                    sentencepair  = mosesfullsentence_processsentence(sentencepair, self.mosesclient, self.experts, self.args.leftcontext, self.args.rightcontext, self.timbloptions, self.args.allornothing)
                elif self.args.mosesX or self.args.mosesW:
                    #weighted score
                    print("(Moses (-X/-W) after classifiers, passing full sentence)",file=sys.stderr)
                    sentencepair = mosesfullsentence_processsentence(sentencepair, self.mosesclient, self.experts, self.args.leftcontext, self.args.rightcontext, self.timbloptions, 0, self.ttable, self.sourceclassencoder, self.targetclassdecoder, self.args.mosestweight)
                else:
                    sentencepair = self.experts.processsentence(sentencepair, self.ttable, self.sourceclassencoder, self.targetclassdecoder, self.args.leftcontext, self.args.rightcontext, self.args.keywords, self.timbloptions, self.lm, self.args.tmweight, self.args.lmweight, None, self.mosesclient)

                return lxml.etree.tostring(sentencepair.xml(), encoding='utf-8',xml_declaration=False, pretty_print=True)
            else:
                request.setHeader(b"content-type", b"text/html")
                return b"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" lang="en">
  <head>
        <meta http-equiv="content-type" content="application/xhtml+xml; charset=utf-8"/>
        <title>Colibrita &cdot; Translation Assistant</title>
  </head>
  <body>
      Enter text in target language, enclose fall-back language content in asteriskes (*):<br />
      <form action="/" method="post">
          <input name="input" /><br />
          <input type="submit">
      </form>
  </body>
</html>"""

    class ColibritaServer:
        def __init__(self, port, experts,ttable, sourceclassencoder, targetclassdecoder, lm, args, timbloptions,mosesclient ):
            assert isinstance(port, int)
            reactor.listenTCP(port, server.Site(ColibritaProcessorResource(experts,ttable ,sourceclassencoder, targetclassdecoder,lm, args, timbloptions, mosesclient)))
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
                if os.path.exists(self.classifiers[classifier].fileprefix + '.ibase'):
                    print("\tClassifier '" + classifier + "' already trained, skipping...", file=sys.stderr)
                    continue

                print("\tTraining '" + classifier + "'", file=sys.stderr)
                self.classifiers[classifier].train()
                self.classifiers[classifier].save()

                #no need to keep it in memory
                self.classifiers[classifier].api = None



    def classify(self, inputfragment, left, right, sentencepair, generalleftcontext, generalrightcontext, generaldokeywords, timbloptions, lm=None,tweight=1,lmweight=1, stats=None):
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
            dist_s = " ".join( ( k + ": " + str(v) for k,v in sorted(distribution.items(),key= lambda x: -1 * x[1])   ) )
            print("\tClassifier translation prior to LM: " + str(inputfragment) + " -> [ DISTRIBUTION:" + dist_s+" ]", file=sys.stderr)
            candidatesentences = []
            bestlmscore = -999999999
            besttscore = -999999999
            besttranslation = "" # best translation WITHOUT LM (only for statistical purposes)
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
                    besttranslation = translation
                candidatesentences.append( ( candidatesentence, outputfragment, tscore, lmscore ) )

            if not stats is None:
                stats['distlength'].append(len(distribution))

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

            if str(outputfragment) != besttranslation:
                stats['lmdifferent'].append( (str(outputfragment), besttranslation) )

            for candidatesentence, targetpattern, tscore, lmscore in candidatesentences:
                if targetpattern != outputfragment:
                    outputfragment.alternatives.append( Alternative( tuple(str(targetpattern).split()), tweight* (tscore-besttscore) + lmweight * (lmscore-bestlmscore) )  )
            print("\tClassifier translation after LM: " + str(inputfragment) + " -> " + str(outputfragment) + " score= " + str(score), file=sys.stderr)

        else:
            outputfragment = Fragment(tuple(classlabel.split()), inputfragment.id, max(distribution.values()))
            for targetpattern, score in distribution.items():
                if targetpattern != classlabel:
                    outputfragment.alternatives.append( Alternative( tuple(targetpattern.split()), score) )
            if not stats is None:
                stats['classifierdistlength'].append(len(distribution))
            dist_s = " ".join( ( k + ": " + str(v) for k,v in sorted(distribution.items(),key= lambda x: -1 * x[1])   ) )
            print("\tClassifier translation " + str(inputfragment) + " -> " + str(outputfragment) + "\t[ DISTRIBUTION:" + dist_s+" ]", file=sys.stderr)

        return outputfragment

    def phrasetablelookup(self, inputfragment, inputfragment_p, sentencepair, targetdecoder, ttable, lm, tweight, lmweight, stats):
        print("\tPhrasetable lookup for '" + str(inputfragment) + "' ...", file=sys.stderr)
        if lm:
            print("\tPhrasetable translation prior to LM: " + str(inputfragment), file=sys.stderr)
            candidatesentences = []
            bestlmscore = -999999999
            besttscore = -999999999
            besttranslation = "" # best translation WITHOUT LM (only for statistical purposes)
            for targetpattern, scores in sorted(ttable[inputfragment_p].items(),key=lambda x: -1* x[1][2]):
                targetpattern_s = targetpattern.tostring(targetdecoder)
                score = scores[2]
                assert score >= 0 and score <= 1
                tscore = math.log(score) #base-e log (LM is converted to base-e upon load)
                translation = tuple(targetpattern_s.split())
                outputfragment = Fragment(translation, inputfragment.id, score)
                candidatesentence = sentencepair.replacefragment(inputfragment, outputfragment, sentencepair.output)
                lminput = " ".join(sentencepair._str(candidatesentence)).split(" ") #joining and splitting deliberately to ensure each word is one item
                lmscore = lm.score(lminput)
                assert lmscore <= 0
                if lmscore > bestlmscore:
                    bestlmscore = lmscore
                if tscore > besttscore:
                    besttscore = tscore
                    besttranslation = translation
                candidatesentences.append( ( candidatesentence, outputfragment, tscore, lmscore ) )

            if not stats is None:
                stats['distlength'].append(len(ttable[inputfragment_p]))

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

            if str(outputfragment) != besttranslation:
                stats['lmdifferent'].append( (str(outputfragment), besttranslation) )

            for candidatesentence, targetpattern, tscore, lmscore in candidatesentences:
                if targetpattern != outputfragment:
                    outputfragment.alternatives.append( Alternative( tuple(str(targetpattern).split()), tweight* (tscore-besttscore) + lmweight * (lmscore-bestlmscore) )  )
            print("\tPhrasetable translation after LM: " + str(inputfragment) + " -> " + str(outputfragment) + " score= " + str(score), file=sys.stderr)

        else:
            for targetpattern, scores in sorted(ttable[inputfragment_p].items(),key=lambda x: -1* x[1][2]):
                targetpattern_s = targetpattern.tostring(targetdecoder)
                outputfragment = Fragment(tuple( targetpattern_s.split(' ') ), inputfragment.id )
                break

            if not stats is None:
                stats['distlength'].append(len(ttable[inputfragment_p]))
            print("\tPhrasetable translation " + str(inputfragment) + " -> " + str(outputfragment) + "\t(out of " + str(len(ttable[inputfragment_p])) +")" , file=sys.stderr)

        return outputfragment




    def processsentence(self, sentencepair, ttable, sourceclassencoder, targetclassdecoder, generalleftcontext, generalrightcontext, generaldokeywords, timbloptions, lm=None,tweight=1,lmweight=1, stats = None, mosesclient=None):
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
                outputfragment =  self.classify(inputfragment, left, right, sentencepair, generalleftcontext, generalrightcontext, generaldokeywords, timbloptions, lm,tweight,lmweight, stats)
                for targetpattern, scores in sorted(ttable[inputfragment_p].items(),key=lambda x: -1* x[1][2]):
                    targetpattern_s = targetpattern.tostring(targetclassdecoder)
                    print("\t(Comparison with best from phrasetable (no LM): " + str(outputfragment) + ": " + " ".join([str(x) for x in scores]) + ")", file=sys.stderr)
                    break
                if stats: stats['classifier'] += 1
                if str(outputfragment) != targetpattern_s:
                    if stats: stats['classifierdifferent'].append( (str(outputfragment), targetpattern_s) )
            elif isinstance(ttable, AlignmentModel) and inputfragment_p and inputfragment_p in ttable:
                outputfragment = self.phrasetablelookup(inputfragment, inputfragment_p, sentencepair, targetclassdecoder, ttable, lm, tweight, lmweight, stats)
                if stats: stats['fallback'] += 1
                if outputfragment is None:
                    raise Exception("No outputfragment found in phrasetable!!! Shouldn't happen")
            elif mosesclient:
                #fall back to moses
                outputfragment = mosesdecode(mosesclient, inputfragment_s, sentencepair, lm, tweight, lmweight, stats)
                if outputfragment is None:
                    #no translation found
                    outputfragment = Fragment(None, inputfragment.id)
                    print("\tNo translation for " + inputfragment_s, file=sys.stderr)
                    if stats: stats['untranslated'] += 1
                else:
                    if stats: stats['fallbackmoses'] += 1
            else:
                #no translation found
                outputfragment = Fragment(None, inputfragment.id)
                print("\tNo translation for " + inputfragment_s, file=sys.stderr)
                if stats: stats['untranslated'] += 1
            sentencepair.output = sentencepair.replacefragment(inputfragment, outputfragment, sentencepair.output)
        return sentencepair



    def loaddttable(self):
        return loaddttable(self.workdir + '/directtranslation.table')

    def initstats(self):
        stats = {}
        stats['untranslated'] = 0
        stats['fallback'] = 0
        stats['classifier'] = 0
        stats['classifierdifferent'] = []
        stats['lmdifferent'] = []
        stats['classifierdistlength'] = []
        stats['distlength'] = []
        stats['fallbackmoses'] = 0

        return stats


    def test(self, data, outputfile, ttable, sourceclassencoder, targetclassdecoder, leftcontext, rightcontext, dokeywords, timbloptions, lm=None,tweight=1,lmweight=1, mosesclient=None):
        stats = self.initstats()
        writer = Writer(outputfile)
        for sentencepair in data:
            sentencepair = self.processsentence(sentencepair, ttable, sourceclassencoder, targetclassdecoder, leftcontext, rightcontext, dokeywords, timbloptions, lm, tweight, lmweight, stats, mosesclient)
            writer.write(sentencepair)
        writer.close()
        if stats:
            self.writestats(stats, outputfile.replace('.xml','') + '.stats')

    def writestats(self, stats, outputfile):
        with open(outputfile,'w',encoding='utf-8') as f:
            totalfragments =  stats['untranslated'] + stats['fallback'] + stats['classifier']
            print("Total fragments:                         " + str(totalfragments) ,file=f)
            if totalfragments > 0:
                print("Untranslated:                            " + str(stats['untranslated']) + " " + str(stats['untranslated'] / totalfragments) ,file=f)
                print("Translated by classifier:                " + str(stats['classifier']) + " " + str(stats['classifier'] / totalfragments) ,file=f)
                print("Translated by phrasetable:               " + str(stats['fallback']) + " " + str(stats['fallback'] / totalfragments) ,file=f)
                print("Classifier made a difference:            " + str(len(stats['classifierdifferent'])) + " " + str(len(stats['classifierdifferent']) / totalfragments) ,file=f)
                if stats['lmdifferent']:
                    print("LM made a difference:                    " + str(len(stats['lmdifferent'])) + " " + str(len(stats['lmdifferent']) / totalfragments) ,file=f)
                if len(stats['classifierdistlength']) > 0:
                    print("Mean length of classifier distribution:  " + str(sum(stats['classifierdistlength']) / len(stats['classifierdistlength']) ) ,file=f)
                if len(stats['distlength']) > 0:
                    print("Mean length of phrasetable distribution: " + str(sum(stats['distlength']) / len(stats['distlength']) ) ,file=f)







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



def setupmosesserver(ttable, sourceclassdecoder, targetclassdecoder, args):
    mosesserverpid = 0
    mosesclient = None
    if args.fallback or args.moses or args.mosesX or args.mosesY or args.mosesW or args.allornothing:

        if isinstance(ttable, str):
            ttablefile = ttable
        else:
            ttablefile = args.output + "/fallback.phrase-table"
            print("Writing " + ttablefile,file=sys.stderr)
            ttable.savemosesphrasetable(ttablefile, sourceclassdecoder, targetclassdecoder)


        print("Writing " + args.output + "/fallback.moses.ini",file=sys.stderr)

        if args.mosestweight:
            tweights = " ".join([ str(x) for x in args.mosestweight])
            lentweights = len(args.mosestweight)
        else:
            tweights = " ".join([ str(x) for x in (0.2,0.2,0.2,0.2)])
            lentweights = 4

        if args.mosesreorderingweight:
            reorderingweights = " ".join([ str(x) for x in args.mosesreorderingweight])
            lenreorderingweights = len(args.mosesreorderingweight)
        else:
            reorderingweights = " ".join([ str(x) for x in (0.3,0.3,0.3,0.3,0.3,0.3)])
            lenreorderingweights = 6



        if args.mosesreorderingmodel:
            reordering = "LexicalReordering name=LexicalReordering0 num-features=" + str(lenreorderingweights) + " type=" + args.mosesreorderingtype + " input-factor=0 output-factor=0 path=" + os.path.abspath(args.mosesreorderingmodel)
            reordering += "\nDistortion"
        else:
            reordering = "Distortion"


        if not args.moseslm and not args.lm:
            raise Exception("You must specify --moseslm or --lm if you use Moses fallback (-F) or --moses!")
        elif args.lm:
            lm = args.lm
        elif args.moseslm:
            lm = args.moseslm

        #write moses.ini
        with open(args.output + '/fallback.moses.ini','w',encoding='utf-8') as f:
            f.write("""
#Moses INI, produced by colibrita.py
[input-factors]
0

[mapping]
0 T 0

[distortion-limit]
6

[feature]
UnknownWordPenalty
WordPenalty
PhrasePenalty
PhraseDictionaryMemory name=TranslationModel0 num-features={lentweights} path={phrasetable} input-factor=0 output-factor=0 table-limit=20
{reordering}
SRILM name=LM0 factor=0 path={lm} order={lmorder}

[weight]
UnknownWordPenalty0= 1
WordPenalty0= {wweight}
PhrasePenalty0= {pweight}
LM0= {lmweight}
TranslationModel0= {tweights}
Distortion0= {dweight}
    """.format(phrasetable=ttablefile,reordering=reordering, lm=lm, lmorder=args.lmorder, lmweight = args.moseslmweight, dweight = args.mosesdweight, tweights=tweights, lentweights=lentweights, wweight=args.moseswweight, pweight = args.mosespweight))
            if args.mosesreorderingmodel:
                f.write("LexicalReordering0= " + reorderingweights + "\n")

        print("Starting Moses Server",file=sys.stderr)
        if args.mosesdir:
            cmd = args.mosesdir + '/bin/mosesserver'
        else:
            cmd = 'mosesserver'
        cmd += ' --serial --server-port ' + str(args.mosesport)
        if args.mosesX or args.mosesY or args.allornothing:
            #do not compete with phrasetable (-X/-Y)
            cmd += " -xml-input exclusive"
        elif args.moses or args.mosesW:
            #compete with phrasetable (-Z/-W)
            cmd += " -xml-input inclusive"
            #inclusive/exclusive makes no difference on non-context informed data
        cmd += ' -f ' + args.output + '/fallback.moses.ini -n-best-list ' + args.output+"/nbest.txt 25"
        print("Calling moses: " + cmd,file=sys.stderr)
        p = subprocess.Popen(cmd.split(),shell=False)
        mosesserverpid = p.pid

        while True:
            time.sleep(5)
            try:
                s = socket.socket()
                s.connect( ("localhost", args.mosesport) )
                break
            except Exception as e:
                print("Waiting for Moses server....", e, file=sys.stderr)

        print("Connecting to Moses Server",file=sys.stderr)
        mosesclient = xmlrpc.client.ServerProxy("http://localhost:" + str(args.mosesport) + "/RPC2")

    return mosesserverpid, mosesclient

def makebaseline(ttable, outputfile, testset,sourceencoder, targetdecoder, mosesclient=None, lm=None,tweight=1, lmweight=1):
    output = Writer(outputfile)
    for sentencepair in testset:
        print("Sentence #" + str(sentencepair.id),file=sys.stderr)
        sentencepair.ref = None
        sentencepair.output = copy(sentencepair.input)
        for left, inputfragment, right in sentencepair.inputfragments():
            inputfragment_s = str(inputfragment)
            print("\tFragment: ", inputfragment_s, file=sys.stderr)
            try:
                inputfragment_p = sourceencoder.buildpattern(inputfragment_s)
            except IOError:
                print("\tNOTICE: One or more words in '" + inputfragment_s + "' were not seen during training",file=sys.stderr)
                inputfragment_p = None

            translation = None
            if inputfragment_p in ttable:
                if lm:
                    candidatesentences = []
                    bestlmscore = -999999999
                    besttscore = -999999999
                    for targetpattern, scores in ttable[inputfragment_p].items():
                        assert scores[2] >= 0 and scores[2] <= 1
                        tscore = math.log(scores[2]) #convert to base-e log (LM is converted to base-e upon load)
                        targetpattern_s = targetpattern.tostring(targetdecoder)
                        outputfragment = Fragment(tuple( targetpattern_s.split(' ') ), inputfragment.id )
                        candidatesentence = sentencepair.replacefragment(inputfragment, outputfragment, sentencepair.output)
                        lminput = " ".join(sentencepair._str(candidatesentence)).split(" ") #joining and splitting deliberately to ensure each word is one item
                        lmscore = lm.score(lminput)
                        assert lmscore <= 0
                        if lmscore > bestlmscore:
                            bestlmscore = lmscore
                        if tscore > besttscore:
                            besttscore = tscore
                        candidatesentences.append( ( candidatesentence, targetpattern, tscore, lmscore ) )
                    #get the strongest sentence
                    maxscore = -9999999999
                    for candidatesentence, targetpattern, tscore, lmscore in candidatesentences:
                        tscore = tweight * (tscore-besttscore)
                        lmscore = lmweight * (lmscore-bestlmscore)
                        score = tscore + lmscore
                        if score > maxscore:
                            maxscore = score
                            translation = targetpattern
                    translation = translation.tostring(targetdecoder)
                else:
                    maxscore = 0
                    for targetpattern, scores in ttable[inputfragment_p].items():
                        targetpattern_s = targetpattern.tostring(targetdecoder)
                        if scores[2] > maxscore:
                            maxscore = scores[2]
                            translation = targetpattern_s
                outputfragment = Fragment(tuple( translation.split(' ') ), inputfragment.id )
                print("\t" + inputfragment_s + " -> " + str(outputfragment), file=sys.stderr)
                sentencepair.output = sentencepair.replacefragment(inputfragment, outputfragment, sentencepair.output)
            elif mosesclient:
                #fall back to moses
                outputfragment = mosesdecode(mosesclient, inputfragment, sentencepair, lm, tweight, lmweight)
                sentencepair.output = sentencepair.replacefragment(inputfragment, outputfragment, sentencepair.output)
            else:
                outputfragment = Fragment(None, inputfragment.id)
                print("\t" + inputfragment_s + " -> NO TRANSLATION", file=sys.stderr)
                sentencepair.output = sentencepair.replacefragment(inputfragment, outputfragment, sentencepair.output)
        output.write(sentencepair)
    testset.close()
    output.close()

def mosesfullsentence_processsentence(sentencepair, mosesclient=None,experts = None,leftcontextsize=0,rightcontextsize=0,timbloptions="", allornothing=0, ttable=None, sourceclassencoder=None,targetclassdecoder=None, tmweights=None):
    if not tmweights:
        tmweights = (0.2,0.2,0.2,0.2)

    sentencepair.ref = None
    sentencepair.output = copy(sentencepair.input)

    print("\tRunning moses decoder (full sentence) for '" + sentencepair.inputstr('*') + "' ...", file=sys.stderr)

    left, inputfragment, right = list(sentencepair.inputfragments())[0] #assume only one inputfragment per sentence
    left = tuple(left.split())
    right = tuple(right.split())
    classifiedfragment = None
    if experts:
        inputfragment_s = str(inputfragment)
        if inputfragment_s in experts.classifiers:
            classifiedfragment =  experts.classify(inputfragment, left, right, sentencepair, leftcontextsize, rightcontextsize, False, timbloptions)



    inputsentence_raw = sentencepair.inputstr("*").strip()
    inputsentence_xml = ""
    leadwords = 0
    tailwords = 0
    havefragment = False
    for word in inputsentence_raw.split(' '):
        if word:
            if word[0] == '*' and word[-1] == '*':
                if classifiedfragment:
                    if havefragment:
                        #already done, ignore remaining words (code only runs on the first encountered word of the fragment!)
                        pass
                    else:
                        #first word of fragment, insertion point is here


                        if isinstance(ttable,AlignmentModel): #(-X option)
                            try:
                                inputfragment_p = sourceclassencoder.buildpattern(inputfragment_s)
                            except IOError:
                                print("\tNOTICE: One or more words in '" + inputfragment_s + "' were not seen during training",file=sys.stderr)
                                inputfragment_p = None
                            pass


                        translation = " ".join(classifiedfragment.value)
                        translations = [ translation ]
                        if allornothing:
                            if classifiedfragment.confidence >= allornothing:
                                print("*** All-or-nothing threshold passed for '" + translation + "', passing as winner to Moses ***",file=sys.stderr)
                                probs = [str(1)] #pass to moses with full confidence
                            else:
                                #don't pass any translation, let Moses handle it completely
                                word = word[1:-1]
                                havefragment = True
                                inputsentence_xml += inputfragment_s + "<wall/>"
                                continue

                        elif isinstance(ttable,AlignmentModel):
                            #(-X option)
                            if inputfragment_p in ttable:
                                #lookup score in phrasetable, replace p(t|s) with classifier score, and compute log linear combination

                                scores = None
                                for targetpattern, tmpscores in sorted(ttable[inputfragment_p].items(),key=lambda x: -1* x[1][2]):
                                    targetpattern_s = targetpattern.tostring(targetclassdecoder)
                                    if targetpattern_s == translation: #bit cumbersome and inefficient but we don't need an encoder this way
                                        scores = tmpscores

                                if scores:
                                    try:
                                        origscore = tmweights[0] * math.log(scores[0]) + tmweights[1] * math.log(scores[1]) + tmweights[2] * math.log(scores[2]) + tmweights[3] * math.log(scores[3])
                                        score = tmweights[0] * math.log(scores[0]) + tmweights[1] * math.log(scores[1]) + tmweights[2] * math.log(classifiedfragment.confidence) + tmweights[3] * math.log(scores[3])
                                    except ValueError: #math domain error
                                        print("WARNING: One of the scores in score vector (or weights) is zero!!",file=sys.stderr)
                                        print("weights: ", tmweights,file=sys.stderr)
                                        print("original scores: ", scores,file=sys.stderr)
                                        score = origscore = -999
                                    score =  math.e ** score
                                    origscore = math.e ** origscore
                                    print("Score for winning target '" + translation + "', classifier=" + str(classifiedfragment.confidence) + ", phrasetable(t|s)=" + str(scores[2]) + ", total(class)=" + str(score), ", total(orig)=" + str(origscore),file=sys.stderr)
                                else:
                                    score = origscore = math.e ** -999
                                    print("**** ERROR ***** Target fragment not found in phrasetable, skipping and ignoring!!! source=" + inputfragment_s + ", target=" + targetpattern_s, file=sys.stderr)

                                probs = [ str(score) ]
                            else:
                                raise Exception("Source fragment not found in phrasetable, shouldn't happen at this point: " + inputfragment_s)
                        else:
                            #(-Z option, args.moses)
                            probs = [ str(classifiedfragment.confidence) ]


                        if not allornothing:
                            for alternative in classifiedfragment.alternatives:
                                translation = " ".join(alternative.value)
                                translations.append( translation )
                                if isinstance(ttable,AlignmentModel):
                                    #(-X option)
                                    if inputfragment_p in ttable:
                                        #lookup score in phrasetable, replace p(t|s) with classifier score, and compute log linear combination

                                        scores = None
                                        for targetpattern, tmpscores in sorted(ttable[inputfragment_p].items(),key=lambda x: -1* x[1][2]):
                                            targetpattern_s = targetpattern.tostring(targetclassdecoder)
                                            if targetpattern_s == translation: #bit cumbersome and inefficient but we don't need an encoder this way
                                                scores = tmpscores

                                        if scores:
                                            try:
                                                origscore = tmweights[0] * math.log(scores[0]) + tmweights[1] * math.log(scores[1]) + tmweights[2] * math.log(scores[2]) + tmweights[3] * math.log(scores[3])
                                                score = tmweights[0] * math.log(scores[0]) + tmweights[1] * math.log(scores[1]) + tmweights[2] * math.log(alternative.confidence) + tmweights[3] * math.log(scores[3])
                                            except ValueError:
                                                print("WARNING: One of the scores in score vector (or weights) is zero!!",file=sys.stderr)
                                                print("weights: ", tmweights,file=sys.stderr)
                                                print("original scores: ", scores,file=sys.stderr)
                                                score = origscore = -999
                                            score = math.e ** score
                                            origscore = math.e ** origscore
                                            print("Score for alternative target '" + translation + "', classifier=" + str(alternative.confidence) + ", phrasetable(t|s)=" + str(scores[2]) + ", total(class)=" + str(score), ", total(orig)=" + str(origscore),file=sys.stderr)
                                        else:
                                            score = origscore = math.e ** -999
                                            print("**** ERROR ***** Target fragment not found in phrasetable, skipping and ignoring!!! source=" + inputfragment_s + ", target=" + targetpattern_s, file=sys.stderr)

                                        probs.append(str(score))
                                    else:
                                        raise Exception("Source fragment not found in phrasetable, shouldn't happen at this point: " + inputfragment_s)

                                else:
                                    #(-Z option, args.moses)
                                    probs.append( str(alternative.confidence) )


                        #Moses XML syntax for multiple options (ugly XML-abuse but okay)
                        translations = "||".join(translations).replace("\"","&quot")
                        probs = "||".join(probs)

                        inputsentence_xml += "<f translation=\"" + translations + "\" prob=\"" + probs + "\">" + inputfragment_s + "</f><wall/>"
                        havefragment = True
                else:
                    #don't pass any translation, let Moses handle it completely
                    word = word[1:-1]
                    havefragment = True
                    inputsentence_xml += word + "<wall/>"
            else:
                inputsentence_xml += "<w translation=\"" + word.replace("\"","&quot;") + "\">" + word + "</w><wall/>"
                if havefragment:
                    tailwords += 1
                else:
                    leadwords += 1


    print("\tMoses input: " + inputsentence_xml.strip(), file=sys.stderr)
    params = {"text":inputsentence_xml.strip(), "align":"false", "report-all-factors":"false", 'nbest':25}
    mosesresponse = mosesclient.translate(params)

    outputsentence = ' '.join([ x.strip() for x in mosesresponse['text'].split(' ') if x.strip() ])
    print("\tMoses response: " + outputsentence + " [leadwords="+str(leadwords) + ":tailwords=" + str(tailwords) +"]", file=sys.stderr)
    outputfragment = Fragment( tuple(outputsentence.split(' ')[leadwords:-tailwords]) , 1 )

    print("\tMoses translation (via full sentence)" + str(inputfragment) + " -> " + str(outputfragment) , file=sys.stderr)

    sentencepair.output = sentencepair.replacefragment(inputfragment, outputfragment, sentencepair.output)
    return sentencepair



def mosesfullsentence(outputfile, testset, mosesclient=None,experts = None,leftcontextsize=0,rightcontextsize=0,timbloptions="", allornothing=0, ttable=None, sourceclassencoder=None,targetclassdecoder=None, tmweights=None):
    output = Writer(outputfile)
    if not tmweights:
        tmweights = (0.2,0.2,0.2,0.2)

    for sentencepair in testset:
        print("Sentence #" + str(sentencepair.id),file=sys.stderr)
        sentencepair = mosesfullsentence_processsentence(sentencepair, mosesclient, experts, leftcontextsize, rightcontextsize, timbloptions, allornothing, ttable, sourceclassencoder, targetclassdecoder, tmweights)
        output.write(sentencepair)
    testset.close()
    output.close()


def mosesdecode(mosesclient, inputfragment, sentencepair, lm, tweight, lmweight, stats=None):
    print("\tRunning moses decoder for '" + str(inputfragment) + "' ...", file=sys.stderr)
    if not inputfragment:
        raise ValueError("No inputfragment specified!")
    params = {"text":str(inputfragment), "align":"false", "report-all-factors":"false", 'nbest':25}
    mosesresponse = mosesclient.translate(params)

    if lm:
        candidatesentences = []
        bestlmscore = -999999999
        besttscore = -999999999
        ceiling = 0

        for nbestitem in mosesresponse['nbest']:
            targetpattern_s = nbestitem['hyp'].strip()
            score = nbestitem['totalScore']
            if not ceiling:
                ceiling = score
            tscore = math.log(score / ceiling) #base-e log (LM is converted to base-e upon load)
            translation = tuple(targetpattern_s.split())
            outputfragment = Fragment(translation, 1, score)
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
        print("\tPhrasetable translation after LM: " + str(inputfragment) + " -> " + str(outputfragment) + " score= " + str(score), file=sys.stderr)

    else:
        targetpattern_s = mosesresponse['text']
        outputfragment = Fragment(tuple( targetpattern_s.split(' ') ), 1 )

        print("\tMoses translation " + str(inputfragment) + " -> " + str(outputfragment) , file=sys.stderr)

    return outputfragment

def main():
    parser = argparse.ArgumentParser(description="Colibrita - Translation Assistance", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--trainfromset',type=str,help="Training mode using pregenerated training set (use with -f)", action='store',default="")
    parser.add_argument('--train',help="Build classifiers and train from scratch based on a parallel corpus", action='store_true')
    parser.add_argument('--source', type=str,help="Source language corpus for training (plaintext)", action='store',required=False)
    parser.add_argument('--target', type=str,help="Target language corpus for training (plaintext)", action='store',required=False)
    parser.add_argument('-M','--phrasetable', type=str,help="Moses phrasetable to use for training (--train), or for testing with moses using -Z/-Y, must be an absolutrie path", action='store',default="")
    parser.add_argument('--trainfortest',type=str, help="Do only limited training that covers a particular test set (speeds up training and reduces memory considerably!), use with --train or --trainfromset", action='store',default="")
    parser.add_argument('--test',type=str,help="Test mode (against a specific test set)", action='store',default="")
    parser.add_argument('--baseline', type=str,help="Baseline (against specified test set)", action='store',default="")
    parser.add_argument('--run',help="Run mode (reads input from stdin)", action='store_true')
    parser.add_argument('--server', help="Server mode (RESTFUL HTTP Server)", action='store_true')
    #parser.add_argument('--igen',dest='settype',help="Instance generation from a training set (-f) without actual training", action='store_const',const='igen')
    parser.add_argument('--debug','-d', help="Debug", action='store_true', default=False)
    parser.add_argument('-a','--autoconf', help="Automatically determine best feature configuration per expert (cross-validated), values for -l and -r are considered maxima, set -k to consider keywords, needs to be specified both at training time and at test time!", action='store_true',default=False)
    parser.add_argument('-l','--leftcontext',type=int, help="Left local context size", action='store',default=0)
    parser.add_argument('-r','--rightcontext',type=int,help="Right local context size", action='store',default=0)

    parser.add_argument('--maxlength',type=int,help="Maximum length of phrases", action='store',default=10)
    parser.add_argument('-k','--keywords',help="Add global keywords in context", action='store_true',default=False)
    parser.add_argument('-A','--allornothing',type=float,help="All or nothing mode. If the winning classifier score exceeds the specified threshold, the translation will be passed to Moses with perfect confidence, otherwise no translation is passed and the SMT decoder resolves it by itself. Passes full sentences through Moses server using XML input (will start a moses server, requires --moseslm)", action='store',default=0.0)
    parser.add_argument('-Z','--moses',help="Pass full sentences  hrough Moses server using XML input (will start a moses server, requires --moseslm). Relies fully on Moses for LM, optional classifier output (if -l,-r) is passed to Moses and competes with phrase-table. Classifier score is the sole score used.", action='store_true',default=False)
    parser.add_argument('-Y','--mosesY',help="Pass full sentences through Moses server using XML input (will start a moses server, requires --moseslm). Relies fully on Moses for LM, optional classifier output (if -l,-r) is passed to Moses and competes with phrase-table. Classifier score is the sole score used.", action='store_true',default=False)
    parser.add_argument('-X','--mosesX',help="Pass full sentences through Moses server using XML input (will start a moses server, requires --moseslm). Relies fully on Moses for LM, optional classifier output (if -l,-r) is passed to Moses but does not compete with phrase-table. Classifier score is integrated in phrasetable using replace method is used for scoring, --mosestweights for weights", action='store_true',default=False)
    parser.add_argument('-W','--mosesW',help="Pass full sentences through through Moses server using XML input (will start a moses server, requires --moseslm). Relies fully on Moses for LM, optional classifier output (if -l,-r) is passed to Moses and competes with phrase-table. Classifier score is integrated in phrasetable using replace method", action='store_true',default=False)
    parser.add_argument('-F','--fallback',help="Attempt to decode unknown fragments using moses (will start a moses server, requires --moseslm or --lm). This is a more constrained version of falling back to Moses only for unknown fragments", action='store_true',default=False)
    parser.add_argument("--kt",dest="bow_absolute_threshold", help="Keyword needs to occur at least this many times in the context (absolute number)", type=int, action='store',default=3)
    parser.add_argument("--kp",dest="bow_prob_threshold", help="minimal P(translation|keyword)", type=float, action='store',default=0.001)
    parser.add_argument("--kg",dest="bow_filter_threshold", help="Keyword needs to occur at least this many times globally in the entire corpus (absolute number)", type=int, action='store',default=20)
    parser.add_argument("--ka",dest="compute_bow_params", help="Attempt to automatically compute --kt,--kp and --kg parameters", action='store_false',default=True)
    #parser.add_argument('-O', dest='timbloptions', help="Timbl Classifier options", type=str,action='store',default="-k 1")
    parser.add_argument('--Tk', dest='timbl_k', help="Timbl k", type=int,action='store',default=1)
    parser.add_argument('--Tclones', dest='timbl_clones', help="Timbl clones (number of CPUs to use for parallel processing)", type=int,action='store',default=1)
    parser.add_argument('-o','--output',type=str,help="Output prefix", required = True )
    #parser.add_argument('--moses', help="Use Moses as Translation Model, no classifiers will be used, use with -T", action='store_true',default=False)
    parser.add_argument('--lm',type=str, help="Use language model in testing (file in ARPA format, as produced by for instance SRILM)", action='store',default="")
    parser.add_argument('--lmorder', type=int, help="Language Model order", action="store", default=3, required=False)
    parser.add_argument('--lmweight',type=float, help="Language model weight (when --lm is used, not for moses)", action='store',default=1)
    parser.add_argument('--tmweight',type=float, help="Translation model weight (when --lm is used, not for moses)", action='store',default=1)
    parser.add_argument('--mosesweights', type=str, help="Read Moses weights from the specified moses.ini file (only reads weights, won't read phrasetable nor reordering model!)", action="store", default="", required=False)
    parser.add_argument('--moseslmweight', type=float, help="Language Model weight for Moses fallback (-F)", action="store", default=0.5, required=False)
    parser.add_argument('--mosesdweight', type=float, help="Distortion Model weight for Moses fallback (-F)", action="store", default=0.3, required=False)
    parser.add_argument('--moseswweight', type=float, help="Word penalty weight for Moses fallback (-F)", action="store", default=-1, required=False)
    parser.add_argument('--mosestweight', type=float, help="Translation Model weight for Moses fallback (-F) (may be specified multiple times for each score making up the translation model)", action="append", required=False)
    parser.add_argument('--mosespweight', type=float, help="Phrase penalty for Moses fallback (-F)", default=0.2, action="store", required=False)
    parser.add_argument('--mosesreorderingmodel', type=str,default="", action="store", required=False)
    parser.add_argument('--mosesreorderingtype', type=str,default="wbe-msd-bidirectional-fe-allff", action="store", required=False)
    parser.add_argument('--mosesreorderingweight', type=float, help="May be specified multiple times for each score making up the reordering model", action="append", required=False)
    parser.add_argument('--port',type=int, help="Server port (use with --server)", action='store',default=7893)
    parser.add_argument('--folds',type=int, help="Number of folds to use for cross-validation (used with -a)", action='store',default=10)
    parser.add_argument('-T','--ttable', type=str,help="Phrase translation table (file) to use, must be a Colibri alignment model (use colibri-mosesphrasetable2alignmodel). Will be tried as a fallback when no classifiers are made, also required when testing with --lm and without classifier training, and when using --trainfromscratch", action='store',default="")

    #setgen options
    parser.add_argument('-p', dest='joinedprobabilitythreshold', help="Used with --trainfromscratch: Joined probabiity threshold for inclusion of fragments from phrase translation-table: min(P(s|t) * P(t|s))", type=float,action='store',default=0)
    parser.add_argument('-D', dest='divergencefrombestthreshold', help="Used with --trainfromscratch: Maximum divergence from best translation option. If set to 0.8, the only alternatives considered are those that have a joined probability of equal or above 0.8 of the best translation option", type=float,action='store',default=0)

    parser.add_argument('--moseslm',type=str, help="Use language model for moses fallback (or issue --lm to use more widely)", action='store',default="")
    parser.add_argument('--mosesdir',type=str, help="Path to moses (for --trainfromscratch)",action='store',default="")
    parser.add_argument('--mosesport',type=int, help="Port for Moses server (will be started for you), if -F is enabled",action='store',default=8080)
    parser.add_argument('--bindir',type=str, help="Path to external bin dir (path where moses bins are installed, for --trainfromscratch)",action='store',default="/usr/local/bin")

    args = parser.parse_args()


    if not args.train and not args.test and not args.trainfromset and not args.trainfortest and not args.baseline and not args.server:
        print("Specify either --train, --test, --trainfromset, --trainfortst, --baseline, --server")
        sys.exit(2)


    timbloptions = "-vdb -G0 -k " + str(args.timbl_k) #if I add -a 0 explicitly (default anyway)   leave_one_out and crossvalidate won't work!!!
    if args.timbl_clones > 1:
        timbloptions += " --clones=" + str(args.timbl_clones)


    if args.mosesweights:
        with open(args.mosesweights,'r',encoding='utf-8') as f:
            for line in f:
                if line[:15] ==  "PhrasePenalty0=":
                    args.mosespweight = float(line[15:].strip())
                    print("Read weight phrase penalty =", str(args.mosespweight),file=sys.stderr)
                elif line[:13] ==  "WordPenalty0=":
                    args.moseswweight = float(line[13:].strip())
                    print("Read weight word penalty = ", str(args.moseswweight),file=sys.stderr)
                elif line[:12] ==  "Distortion0=":
                    args.mosesdweight = float(line[12:].strip())
                    print("Read weight distortion = ", str(args.mosesdweight),file=sys.stderr)
                elif line[:4] ==  "LM0=":
                    args.moseslmweight = float(line[4:].strip())
                    print("Read weight lm = ", str(args.moseslmweight),file=sys.stderr)
                elif line[:18] ==  "TranslationModel0=":
                    args.mosestweight = [ float(x) for x in line[18:].strip().split() ]
                    print("Read weight tm = ", repr(args.mosestweight),file=sys.stderr)
                elif line[:19] ==  "LexicalReordering0=":
                    args.mosesreorderingweight = [ float(x) for x in line[19:].strip().split() ]
                    print("Read weight reordering = ", repr(args.mosesreorderingweight),file=sys.stderr)


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
        keywordmodelfile  = args.output + '/keywords.colibri.indexedpatternmodel'

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
                cmd = "colibri-mosesphrasetable2alignmodel -i " + args.phrasetable + " -m " + args.output+"/testfragments.colibri.unindexedpatternmodel -o " + args.output + "/colibri.alignmodel -j " + str(args.joinedprobabilitythreshold) + " -d " + str(args.divergencefrombestthreshold) + " -S " + sourceclassfile + " -T " + targetclassfile
                print("2.4) Creating alignment model from Moses phrasetable, constrained by testset: " + cmd,file=sys.stderr)
                r = os.system(cmd)
                if r != 0:
                    print("Failed",file=sys.stderr)
                    sys.exit(2)
        else:
            #not constrained by testset

            if not os.path.exists(args.output+"/colibri.alignmodel"):
                #Convert moses phrasetable to alignment model, unconstrained by testset. No normalisation
                cmd = "colibri-mosesphrasetable2alignmodel -i " + args.phrasetable + " -o " + args.output + "/colibri.alignmodel -j " + str(args.joinedprobabilitythreshold) + " -d " + str(args.divergencefrombestthreshold) + " -S " + sourceclassfile + " -T " + targetclassfile
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
            #threshold 1 is needed here
            cmd = "colibri-patternmodeller -t 1 -l " + str(args.maxlength) + " -f " + targetcorpusfile + " -o " + targetmodelfile + " -j " + args.output + "/targetconstraints.colibri.patternsetmodel"
            print("3.3) Building indexed pattern model on target corpus (constrained by phrasetable): ", cmd, file=sys.stderr)
            r = os.system(cmd)
            if r != 0:
                print("Failed",file=sys.stderr)
                sys.exit(2)

        if args.keywords and not os.path.exists(keywordmodelfile):
            cmd = "colibri-patternmodeller -t " + str(max(args.bow_filter_threshold,args.bow_absolute_threshold)) + " -l 1 -f " + targetcorpusfile + " -o " + keywordmodelfile
            print("3.3) Building keyword model: indexed pattern model on target corpus: ", cmd, file=sys.stderr)
            r = os.system(cmd)
            if r != 0:
                print("Failed",file=sys.stderr)
                sys.exit(2)


        if not os.path.exists(args.output+'/classifier.conf'):
            if args.keywords:
                keywordopts = "-k --kt " + str(args.bow_absolute_threshold) + " --kp " + str(args.bow_prob_threshold) + " --kg " + str(args.bow_filter_threshold) + " --km " + keywordmodelfile
            else:
                keywordopts = ""
            cmd = "colibri-extractfeatures --crosslingual " + keywordopts + " -C -X -i " + args.output + "/colibri.alignmodel -f " + targetcorpusfile + " -l " + str(args.leftcontext) + " -r " + str(args.rightcontext) + " -o " + args.output + " -s " + sourcemodelfile + " -t " + targetmodelfile + " -S " + sourceclassfile + " -T " + targetclassfile + " -c " + targetclassfile #TODO: do sourcemodel and targetmodel work with --crosslingual??
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


    #if args.fallback:
    #    mosesserver = xmlrpc.client.ServerProxy("http://localhost:" + str(args.mosesport) + "/RPC2")
    #else:
    #    mosesserver = None

    if args.lm and not args.moses and not args.mosesX and not args.mosesY and not args.mosesW and not args.allornothing:
        print("Loading Language model " + args.lm, file=sys.stderr)
        lm = ARPALanguageModel(args.lm)
    else:
        lm = None


    if (args.fallback or args.moses or args.mosesX or args.mosesY or args.mosesW or args.allornothing) and args.test and not args.baseline and not args.leftcontext and not args.rightcontext:
        # --test -F without any context  is the same as --baseline -F
        args.baseline = args.test
        args.test = ""

    if args.baseline or args.test or args.server or args.run:
        if not os.path.isdir(args.output):
            print("Output directory " + args.output + " does not exist, did you forget to train the system first?", file=sys.stderr)
            sys.exit(2)
        if not os.path.exists(args.output + "/colibri.alignmodel") and not (args.phrasetable and (args.moses or args.mosesY)):
            print("Alignment model in output directory " + args.output + " does not exist, did you forget to train the system first?", file=sys.stderr)
            sys.exit(2)
        if not os.path.exists(args.output+'/colibrita.conf'):
            print("No colibrita.conf found in specified directory (-o). Has the system been trained?", file=sys.stderr)
            sys.exit(2)

        print("Loading configuration from " + args.output + "/colibrita.conf", file=sys.stderr)
        conf = pickle.load(open(args.output + '/colibrita.conf','rb'))
        sourceclassfile = conf['sourceclassfile']
        targetclassfile = conf['targetclassfile']

        print("Loading source class encoder " + sourceclassfile, file=sys.stderr)
        sourceclassencoder = ClassEncoder(sourceclassfile)
        print("Loading target class decoder " + targetclassfile, file=sys.stderr)
        targetclassdecoder = ClassDecoder(targetclassfile)

        if args.phrasetable and not args.mosesX and not args.mosesW: #(works for mosesZ and mosesY)
            print("Using moses phrasetable rather than colibri alignment model!" ,file=sys.stderr)
            ttable = args.phrasetable #we use a str instead of AlignmentModel instance indicate we have a moses phrasetable
        else:
            print("Loading translation table " + args.output + "/colibri.alignmodel",file=sys.stderr)
            ttable = AlignmentModel(args.output + "/colibri.alignmodel") #we still use this for --moses mode too, as we save it to moses-style phrasetable ourselves. This phrasetable may be a constrained version of the original!!!

        mosesserverpid, mosesclient = setupmosesserver(ttable, ClassDecoder(sourceclassfile), targetclassdecoder, args)
    else:
        mosesserverpid = 0

    if args.baseline:

        data = Reader(args.baseline)
        print("Making baseline",file=sys.stderr)
        if args.moses or args.mosesY or args.mosesX or args.mosesW or args.allornothing:
            print("(Moses only, passing full sentence)",file=sys.stderr)
            mosesfullsentence(args.output + '.output.xml', data, mosesclient)
        else:
            if args.lm:
                print("(with LM)",file=sys.stderr)
                makebaseline(ttable, args.output + '.output.xml', data, sourceclassencoder, targetclassdecoder, mosesclient, lm, args.tmweight, args.lmweight)
            else:
                makebaseline(ttable, args.output + '.output.xml', data, sourceclassencoder, targetclassdecoder, mosesclient)

    elif args.test:
        print("Parameters: ", repr(args), file=sys.stderr)

        if (args.leftcontext or args.rightcontext or args.keywords): # and not args.moses:

            experts = ClassifierExperts(args.output)
            print("Loading classifiers",file=sys.stderr)
            experts.load(timbloptions, args.leftcontext, args.rightcontext, args.keywords, None, args.autoconf)


            data = Reader(args.test)
            if args.moses or args.mosesY or args.allornothing:
                #classifier score
                print("(Moses (-Z/-Y) after classifiers, passing full sentence)",file=sys.stderr)
                mosesfullsentence(args.output + '.output.xml', data, mosesclient, experts, args.leftcontext, args.rightcontext, timbloptions, args.allornothing)
            elif args.mosesX or args.mosesW:
                #weighted score
                print("(Moses (-X/-W) after classifiers, passing full sentence)",file=sys.stderr)
                mosesfullsentence(args.output + '.output.xml', data, mosesclient, experts, args.leftcontext, args.rightcontext, timbloptions, 0, ttable, sourceclassencoder, targetclassdecoder, args.mosestweight)
            else:
                print("Running...",file=sys.stderr)
                experts.test(data, args.output + '.output.xml', ttable, sourceclassencoder,targetclassdecoder, args.leftcontext, args.rightcontext, args.keywords, timbloptions , lm,  args.tmweight, args.lmweight, mosesclient)

        else:
            print("Don't know what to do! Specify some classifier options or -T with --lm or --baseline", file=sys.stderr)

    elif args.server:
        if (args.leftcontext or args.rightcontext or args.keywords): # and not args.moses:
            experts = ClassifierExperts(args.output)
            print("Loading classifiers",file=sys.stderr)
            experts.load(timbloptions, args.leftcontext, args.rightcontext, args.keywords, None, args.autoconf)
        else:
            experts = None

        print("Starting Colibrita server on port " + str(args.port),file=sys.stderr)
        ColibritaServer(args.port, experts, ttable, sourceclassencoder, targetclassdecoder, lm, args, timbloptions, mosesclient)

    elif args.run:
        print("Reading from standard input, enclose words/phrases in fallback language in asteriskes (*), type q<enter> to quit",file=sys.stderr)
        for line in sys.stdin:
            line = line.strip()
            if line == "q":
                break
            else:
                sentencepair = plaintext2sentencepair(line)
                if experts:
                    sentencepair = experts.processsentence(sentencepair, ttable, sourceclassencoder, targetclassdecoder, args.leftcontext, args.rightcontext, args.keywords, timbloptions, lm, args.tmweight, args.lmweight, None, mosesclient)
                print(str(lxml.etree.tostring(sentencepair.xml(), encoding='utf-8',xml_declaration=False, pretty_print=True),'utf-8'), file=sys.stderr)
                print(sentencepair.outputstr())
    else:
        print("Don't know what to do!", file=sys.stderr)

    if mosesserverpid:
        print("Stopping moses server (" + str(mosesserverpid) + ")", file=sys.stderr)
        os.kill(mosesserverpid, signal.SIGKILL)
        time.sleep(3)

    print("All done.", file=sys.stderr)


if __name__ == '__main__':
    main()
    sys.exit(0)

