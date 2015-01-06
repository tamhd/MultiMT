#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ./tmtriangulate.py combine_given_weights -ps test/model1 -pt test/model2 -o test/phrase-table_sample -t tempdir
#  This class implement a naive method for triangulation: nothing
#  The most important part of this method is to initialize variables

#TODO: Implement a totally new method for computing the probabilities and lexical weights by the occurrences and co-occurrences

from __future__ import division, unicode_literals
import sys
import os
import gzip
import argparse
import copy
import re
from math import log, exp, sqrt
from collections import defaultdict
from operator import mul
from tempfile import NamedTemporaryFile
#from tmcombine import Moses, Moses_Alignment, to_list
from subprocess import Popen


try:
    from itertools import izip
except:
    izip = zip

def parse_command_line():

    parser = argparse.ArgumentParser(description="Combine translation models. Check DOCSTRING of the class Triangulate_TMs() and its methods for a more in-depth documentation and additional configuration options not available through the command line. The function test() shows examples")

    group1 = parser.add_argument_group('Main options')
    group2 = parser.add_argument_group('More model combination options')
    group3 = parser.add_argument_group('Naive triangulation')

    group1.add_argument('action', metavar='ACTION', choices=["combine_given_weights","maximize_given_weights","compute_by_occurrences"],
                    help='What you want to do with the models. One of %(choices)s.')

    group1.add_argument('-ps', metavar='DIRECTORY', dest='srcpvt',
                    help='model of the source and pivot, actually, it is going to be pivot-source')

    group1.add_argument('-pt', metavar='DIRECTORY', dest='pvttgt',
                    help='model of pivot and target')

    group1.add_argument('-w', '--weights', dest='weights', action=to_list,
                    default=None,
                    help='weight vector. Format 1: single vector, one weight per model. Example: \"0.1,0.9\" ; format 2: one vector per feature, one weight per model: \"0.1,0.9;0.5,0.5;0.4,0.6;0.2,0.8\"')

    group1.add_argument('-m', '--mode', type=str,
                    default="interpolate",
                    choices=["counts","interpolate","loglinear"],
                    help='basic mixture-model algorithm. Default: %(default)s. Note: depending on mode and additional configuration, additional statistics are needed. Check docstring documentation of Triangulate_TMs() for more info.')

    group1.add_argument('-i', '--inverted', type=str,
                    choices=['none',"src-pvt","tgt-pvt",'both'],
                    help='choose to invert the phrasetable if you don\'t have two phrase table in the form of pvt-src and pvt-tgt. You may choose to invert one of them or both of them')

    group1.add_argument('-co', '--co-occurrences', dest='computation',
                    default="minimum",
                    choices=['minimum',"maximum","arithmetic-mean",'geometric-mean'],
                    help='choose to measures the co-occurrences if the action is compute_by_occurrences, you have 4 options: minimum, maximum, arithmetic mean and geometric mean')

    group1.add_argument('-r', '--reference', type=str,
                    default=None,
                    help='File containing reference phrase pairs for cross-entropy calculation. Default interface expects \'path/model/extract.gz\' that is produced by training a model on the reference (i.e. development) corpus.')

    group1.add_argument('-o', '--output', type=str,
                    default="-",
                    help='Output file (phrase table). If not specified, model is written to standard output.')

    group1.add_argument('--output-lexical', type=str,
                    default=None,
                    help=('Not only create a combined phrase table, but also combined lexical tables. Writes to OUTPUT_LEXICAL.e2f and OUTPUT_LEXICAL.f2e, or OUTPUT_LEXICAL.counts.e2f in mode \'counts\'.'))

    group1.add_argument('--lowmem', action="store_true",
                    help=('Low memory mode: requires two passes (and sorting in between) to combine a phrase table, but loads less data into memory. Only relevant for mode "counts" and some configurations of mode "interpolate".'))

    group1.add_argument('--tempdir', type=str,
                    default=None,
                    help=('Temporary directory in --lowmem mode.'))

    group1.add_argument('-t', '--tempdir2', dest='tempdir2', type=str,
                    default=None,
                    help=('Temporary directory to put the intermediate phrase'))


    group2.add_argument('--i_e2f', type=int,
                    default=0, metavar='N',
                    help=('Index of p(f|e) (relevant for mode counts if phrase table has custom feature order). (default: %(default)s)'))

    group2.add_argument('--i_e2f_lex', type=int,
                    default=1, metavar='N',
                    help=('Index of lex(f|e) (relevant for mode counts or with option recompute_lexweights if phrase table has custom feature order). (default: %(default)s)'))

    group2.add_argument('--i_f2e', type=int,
                    default=2, metavar='N',
                    help=('Index of p(e|f) (relevant for mode counts if phrase table has custom feature order). (default: %(default)s)'))

    group2.add_argument('--i_f2e_lex', type=int,
                    default=3, metavar='N',
                    help=('Index of lex(e|f) (relevant for mode counts or with option recompute_lexweights if phrase table has custom feature order). (default: %(default)s)'))

    group2.add_argument('--number_of_features', type=int,
                    default=4, metavar='N',
                    help=('Combine models with N + 1 features (last feature is constant phrase penalty). (default: %(default)s)'))

    group2.add_argument('--normalized', action="store_true",
                    help=('for each phrase pair x,y: ignore models with p(y)=0, and distribute probability mass among models with p(y)>0. (default: missing entries (x,y) are always interpreted as p(x|y)=0). Only relevant in mode "interpolate".'))

    group2.add_argument('--write-phrase-penalty', action="store_true",
      help=("Include phrase penalty in phrase table"))

    group2.add_argument('--recompute_lexweights', action="store_true",
                    help=('don\'t directly interpolate lexical weights, but interpolate word translation probabilities instead and recompute the lexical weights. Only relevant in mode "interpolate".'))

    group3.add_argument('--command', '--./tmtriangulate.py combine_given_weights -ps model1 -pt model2 -o output_phrasetable -t tempdir', action="store_true",
                    help=('If you wish to run the naive approach, the command above would work, in which: model1 = pivot-source model, model2 = pivot-target model'))


    return parser.parse_args()

#convert weight vector passed as a command line argument
class to_list(argparse.Action):
     def __call__(self, parser, namespace, weights, option_string=None):
         if ';' in weights:
             values = [[float(x) for x in vector.split(',')] for vector in weights.split(';')]
         else:
             values = [float(x) for x in weights.split(',')]
         setattr(namespace, self.dest, values)


# New configuration of Moses class

class Moses:
    ''' Moses interface for loading/writing models
        It keeps the value of src-pvt word count
    '''
    def __init__(self, number_of_features=4):
        self.number_of_features = number_of_features

        self.word_pairs_e2f = defaultdict(lambda: defaultdict())
        self.word_pairs_f2e = defaultdict(lambda:defaultdict())

        self.phrase_count_e = defaultdict()
        self.phrase_count_f = defaultdict()

    # when you read the alignment, save the count of word here, for example: e2f[src][tgt] = 4, f2e[tgt][src] = 3

    def _print_lexical_file:
        ''' print the lexical file based on word pairs
        '''

#merge the noisy phrase table
class Merge_TM():
    """This class take input as one noisy phrase table in which it consists of so many repeated lines.
       The output of this class should be one final clean phrase table

       The tasks which have to be done are:
        + Merge TM by summing them up
        + Merge TM by taking the maximum
        + Merge TM by co-occurrence
    """
    def __init__(self,model=None,
                      output_file=None,
                      mode='interpolate',
                      lang_src=None,
                      lang_target=None,
                      output_lexical=None,
                      action="compute_by_occurrences",
                      moses_interface=None
                      ):

        self.mode = mode
        self.model = model # the model file
        self.output_file = output_file
        self.lang_src = lang_src
        self.lang_target = lang_target
        self.loaded = defaultdict(int)
        self.output_lexical = output_lexical
        self.action=action
        self.moses_interface=moses_interface


    def _combine_TM(self,flag=False,prev_line=None):
        '''
        Summing up the probability
        Get the unification of alignment
        Get the sum of counts
        '''
        prev_line = []
        sys.stderr.write("\nCombine Multiple lines by option: " + self.action + "\n")
        output_object = handle_file(self.output_file,'open',mode='w')
        sys.stderr.write("Start merging multiple lines ...")

        # define the action
        if (self.action == 'combine_given_weights'):
            self._line_traversal = self._regular_traversal
            self._combine_lines = self._combine_sum
        elif (self.action == 'maximize_given_weights'):
            self._line_traversal = self._regular_traversal
            self._combine_lines = self._combine_max
        elif (self.action == 'compute_by_occurrences'):
            self._line_traversal = self._normalized_traversal
            self._combine_lines = self._combine_occ
        else:
            # by default, let say we take the sum
            self._line_traversal = self._normalized_traversal
            self._combine_lines = self._combine_occ
        self._line_traversal(flag,prev_line,output_object)
        handle_file(self.output_file,'close',output_object,mode='w')

    def _regular_traversal(self,flag=False,prev_line=None,output_object=None):
        ''' Traver through the phrase-table by old way, previous line is the new line
        '''
        count = 0
        for line in self.model:
            # print counting lines
            if not count%100000:
                sys.stderr.write(str(count)+'...')
            count+=1

            line = _load_line(line)
            if (flag):
                if (line[0] == prev_line[0] and line[1] == prev_line[1]):
                    # combine current sentence to previous sentence, return previous sentence
                    prev_line = self._combine_lines(prev_line, line)
                    continue
                else:
                # when you get out of the identical blog, print your previous sentence
                    outline = _write_phrasetable_file(prev_line)
                    output_object.write(outline)
                    prev_line = line
                    flag = False

            elif (prev_line):
                if (line[0] == prev_line[0] and line[1] == prev_line[1]):
                # if you see a second sentence in a block, turn flag to True and combine
                    prev_line = self._combine_lines(prev_line, line)
                    flag = True
                    continue
                else:
                    outline = _write_phrasetable_file(prev_line)
                    output_object.write(outline)
                    prev_line = line
            else:
                # the first position
                prev_line = line
        if (prev_line):
            outline = _write_phrasetable_file(prev_line)
            output_object.write(outline)
        sys.stderr.write("Done\n")

        return None

    def _normalized_traversal(self,flag=False,prev_line=None,output_object=None):
        ''' Traver through the phrase-table to re-compute the phrase by co-occurrences
        '''
        count = 0
        for line in self.model:
            # print counting lines
            if not count%100000:
                sys.stderr.write(str(count)+'...')
            count+=1

            line = _load_line(line)
            line[4][0] = self.moses_interface.phrase_count_f[line[1]] # target
            line[4][1] = self.moses_interface.phrase_count_e[line[0]] # source
            if (prev_line):
                if (line[0] == prev_line[0] and line[1] == prev_line[1]):
                    # combine current sentence to previous sentence, return previous sentence
                    prev_line = self._combine_lines(prev_line, line)
                    flag = True
                    continue
                else:
                    # when you get out of the identical blog, print your previous sentence
                    prev_line = self._recompute_occ(prev_line)
                    outline = _write_phrasetable_file(prev_line)
                    output_object.write(outline)
                    prev_line = line
                    flag = False
            else:
                # the first position
                prev_line = line
        if (prev_line):
            outline = _write_phrasetable_file(prev_line)
            output_object.write(outline)
        sys.stderr.write("Done\n")

    def _recompute_occ(self,line):
        '''
        Compute the value of a single according to the co-occurrence
        format: src ||| tgt ||| prob1 lex1 prob2 lex2 ||| align ||| c_t c_s c_s_t ||| |||
        '''
        coocc = line[4][2]
        count_s = line[4][1]
        count_t = line[4][0]

        # probability
        if (coocc == 0 and count_t == 0):
            line[2][0] = 0
        else:
            line[2][0] = coocc/count_t # p(s|t)
        if (coocc == 0 and count_s == 0):
            line[2][2] = 0
        else:
            line[2][2] = coocc/count_s # p(t|s)
        return line

    def _combine_occ(self,prev_line=None,cur_line=None):
        '''
        Calculate the value of combine occ by the co-occurrence
        rather than the probabilities
        '''
        # probability
        for i in range(4):
            prev_line[2][i] += cur_line[2][i]
        # alignment
        for src,key in cur_line[3].iteritems():
            for tgt in key:
                if (tgt not in prev_line[3][src]):
                    prev_line[3][src].append(tgt)
        # count
        prev_line[4][0] = self.moses_interface.phrase_count_f[prev_line[1]] # target
        prev_line[4][1] = self.moses_interface.phrase_count_e[prev_line[0]] # source
        prev_line[4][2] += cur_line[4][2]
        return prev_line

    def _combine_sum(self,prev_line=None,cur_line=None):
        '''
        Summing up the probability
        Get the unification of alignment
        Get the sum of counts
        '''
        # probability
        for i in range(4):
            prev_line[2][i] += cur_line[2][i]
        # alignment
        for src,key in cur_line[3].iteritems():
            for tgt in key:
                if (tgt not in prev_line[3][src]):
                    prev_line[3][src].append(tgt)
        # count
        if (cur_line[4][0] != prev_line[4][0] or cur_line[4][1] != prev_line[4][1]):
            sys.exit(1)
        else:
            prev_line[4][2] += cur_line[4][2]
        return prev_line

    def _combine_max(self,prev_line=None,cur_line=None):
        '''
        Get the maximum the probability
        Get the unification of alignment
        Get the sum of counts
        '''
        # probability
        for i in range(4):
            prev_line[2][i] = max(prev_line[2], cur_line[2][i])
        # alignment
        for src,key in cur_line[3].iteritems():
            for tgt in key:
                if (tgt not in prev_line[3][src]):
                    prev_line[3][src].append(tgt)
        # count
        if (cur_line[4][0] != prev_line[4][0] or cur_line[4][1] != prev_line[4][1]):
            sys.exit(1)
        else:
            prev_line[4][2] += cur_line[4][2]
        return prev_line



class Triangulate_TMs():
    """This class handles the various options, checks them for sanity and has methods that define what models to load and what functions to call for the different tasks.
       Typically, you only need to interact with this class and its attributes.

    """

    #some flags that change the behaviour during scoring. See init docstring for more info
    flags = {'normalized':False,
            'recompute_lexweights':False,
            'intersected_cross-entropies':False,
            'normalize_s_given_t':None,
            'normalize-lexical_weights':True,
            'add_origin_features':False,
            'write_phrase_penalty':False,
            'lowmem': False,
            'i_e2f':0,
            'i_e2f_lex':1,
            'i_f2e':2,
            'i_f2e_lex':3
            }

    # each model needs a priority. See init docstring for more info
    _priorities = {'primary':1,
                    'map':2,
                    'supplementary':10}

    def __init__(self,model1=None,
                      model2=None,
                      weights=None,
                      output_file=None,
                      mode='interpolate',
                      inverted=None,
                      action=None,
                      computed=None,
                      tempdir=None,
                      number_of_features=4,
                      lang_src=None,
                      lang_target=None,
                      output_lexical=None,
                      **flags):

        self.mode = mode
        self.output_file = output_file
        self.lang_src = lang_src
        self.lang_target = lang_target
        self.loaded = defaultdict(int)
        self.output_lexical = output_lexical
        self.flags = copy.copy(self.flags)
        self.flags.update(flags)
        self.inverted = inverted
        self.tempdir=tempdir
        self.flags['i_e2f'] = int(self.flags['i_e2f'])
        self.flags['i_e2f_lex'] = int(self.flags['i_e2f_lex'])
        self.flags['i_f2e'] = int(self.flags['i_f2e'])
        self.flags['i_f2e_lex'] = int(self.flags['i_f2e_lex'])
        # Variable 'mode' is preserved to prepare for multiple way of trianuglating.
        # At this moment, it is  interpolate
        if mode not in ['interpolate']:
            sys.stderr.write('Error: mode must be either "interpolate", "loglinear" or "counts"\n')
            sys.exit(1)

        #models,number_of_features = self._sanity_checks(models,number_of_features)
        number_of_features = int(number_of_features)
        self.model1=model1
        self.model2=model2
        self.action = action
        self.computed = get_minimum_counts
        if (self.action == 'compute_by_occurrences'):
            if (computed == 'maximum'):
                self.computed = get_maximum_counts
            elif(computed == 'arithmetic-mean'):
                self.computed = get_arithmetic_mean
            elif(computed == 'geometric-mean'):
                self.computed = get_geometric_mean

        # The model to keep word count
        self.moses_interface = Moses(4)

    def _sanity_checks(self,models,number_of_features):
        """check if input arguments make sense
           this function is important in TMCombine
           TODO: Think how to use this function in triangulation, which feature is necessary to check
        """
        #Note: This is a function which I borrow from TMCombine, however, it has not been used at all :)
        return None

    def combine_standard(self,weights=None):
        """write a new phrase table, based on existing weights of two other tables
           #NOTE: Indeed, all processes start here"""
        data = []

        if self.mode == 'interpolate':
            if self.flags['recompute_lexweights']:
                data.append('lexical')
            if self.flags['normalized'] and self.flags['normalize_s_given_t'] == 't' and not self.flags['lowmem']:
                data.append('pt-target')

        if self.flags['lowmem'] and (self.mode == 'counts' or self.flags['normalized'] and self.flags['normalize_s_given_t'] == 't'):
            self._inverse_wrapper(weights,tempdir=self.flags['tempdir'])
        else:
            file1obj = handle_file(os.path.join(self.model1,'model','phrase-table'), 'open', 'r')
            file2obj = handle_file(os.path.join(self.model2,'model','phrase-table'), 'open', 'r')
            model1 = (file1obj, 1, 1)
            model2 = (file2obj, 1, 2)
            model1, model2 = self._ensure_inverted(model1, model2)
            output_object = handle_file(self.output_file,'open',mode='w')
            self._write_phrasetable(model1, model2, output_object)
            handle_file(self.output_file,'close',output_object,mode='w')

    def _ensure_inverted(self, model1, model2):
        ''' make sure that all the data is in the right format
        '''
        # do nothing for inverted
        if (not self.inverted):
            return (model1, model2)

        models=[]
        if (self.inverted == 'src-pvt'):
            models.append(model1)
        elif (self.inverted == 'tgt-pvt'):
            models.append(model2)
        elif (self.inverted == 'both'):
            models.append(model1)
            models.append(model2)
        else:
            # self.inverted = none or whatever
            return (model1, model2)

        for mod in models:
            outfile = NamedTemporaryFile(delete=False,dir=self.tempdir)
            output_contr = handle_file(outfile.name, 'open', mode='w')
            print "Inverse model ", mod[0], " > ", outfile.name
            #Read line, revert the data to pvt ||| X ||| prob ||| align ||| count ||| |||
            count=0
            for line in mod[0]:
                if not count%100000:
                    sys.stderr.write(str(count)+'...')
                count+=1

                line = _load_line(line)
                # reversing
                pvt_word = line[1].strip()
                line[1] = line[0].strip()
                line[0] = pvt_word
                # reverse probability
                features = line[2]
                tmp = features[0]
                features[0] = features[2]
                features[2] = tmp
                tmp = features[1]
                features[1] = features[3]
                features[3] = tmp

                # reverse alignment
                phrase_align = defaultdict(lambda: []*3)
                for s,t_list in line[3].iteritems():
                    for t in t_list:
                        phrase_align[t].append(s)
                # break the count
                # sometimes, the count is too big
                if (len(line[4]) > 1):
                    tmp = line[4][0]
                    line[4][0] = line[4][1]
                    line[4][1] = tmp
                outline = _write_phrasetable_file(line[0], line[1], features, phrase_align, line[4])
                output_contr.write(outline)
            handle_file(outfile,'close',output_contr,mode='w')
            tmpfile = sort_file(outfile.name,tempdir=self.tempdir)
            #TODO: Check if it make senses
            if (mod[2] == model1[2]):
                model1 = (tmpfile, model1[1], model1[2])
            elif (mod[2] == model2[2]):
                model2 = (tmpfile, model2[1], model2[2])
        print "finish reversing"
        return (model1, model2)


    def _phrasetable_traversal(self,model1,model2,prev_line1,prev_line2,deci,output_object,iteration):
        ''' A non-recursive way to read two model
            Notes: In moses phrase table, the longer phrase appears earlier than the short phrase
        '''
        line1 =  _load_line(model1[0].readline())
        line2 =  _load_line(model2[0].readline())
        count = 0
        while(1):
            if not count%100000:
                sys.stderr.write(str(count)+'...')
            count+=1
            if (self.phrase_equal[0]):
                if (line1 and line1[0] == self.phrase_equal[0]):
                    self.phrase_equal[1].append(line1)
                    line1 =  _load_line(model1[0].readline())
                    continue
                elif (line2 and line2[0] == self.phrase_equal[0]):
                    self.phrase_equal[2].append(line2)
                    line2 = _load_line(model2[0].readline())
                    continue
                else:
                    self._combine_and_print(output_object)

            # handle if the matching is found
            if (not line1 or not line2):
                #self.phrase_equal = defaultdict(lambda: []*3)
                self._combine_and_print(output_object)
                sys.stderr.write("Finish loading\n")
                return None

            # handle if the machine is not found
            if (not self.phrase_equal[0]):
                if (line1[0] == line2[0]):
                    self.phrase_equal[0] = line1[0]
                elif (line1[0].startswith(line2[0])):
                    line1 = _load_line(model1[0].readline())
                elif (line2[0].startswith(line1[0])):
                    line2 = _load_line(model2[0].readline())
                elif (line1[0] < line2[0]):
                    line1 = _load_line(model1[0].readline())
                elif (line1[0] > line2[0]):
                    line2 = _load_line(model2[0].readline())

    def _combine_and_print(self,output_object):
        ''' Follow Cohn at el.2007
        The conditional over the source-target pair is: p(s|t) = sum_i p(s|i,t)p(i|t) = sum_i p(s|i)p(i|t)
        in which i is the pivot which could be found in model1(pivot-src) and model2(src-tgt)
        After combining two phrase-table, print them right after it
        '''
        #if (len(self.phrase_equal[1]) > 10 and len(self.phrase_equal[2]) > 10):
            #print "Huge match: ", self.phrase_equal[1][0][0], len(self.phrase_equal[1]), len(self.phrase_equal[2])
        for phrase1 in self.phrase_equal[1]:
            for phrase2 in self.phrase_equal[2]:
                if (phrase1[0] != phrase2[0]):
                    sys.exit("THE PIVOTS ARE DIFFERENT")
                src, tgt = phrase1[1], phrase2[1]

                #self.phrase_probabilities=[0]*4
                # A-B = A|B|P(A|B) L(A|B) P(B|A) L(B|A)
                # A-C = A|C|P(A|C) L(A|C) P(C|A) L(C|A)
                ## B-C = B|C|P(B|C) L(B|C) P(C|B) L(C|B)

                features = self._get_features_Cohn(src, tgt, phrase1[2], phrase2[2])
                word_alignments = self._get_word_alignments(src, tgt, phrase1[3], phrase2[3])
                word_counts = self._get_word_counts(src, tgt, phrase1[4], phrase2[4])
                outline = _write_phrasetable_file([src,tgt,features,word_alignments,word_counts])
                output_object.write(outline)
                self._update_moses(src,tgt,word_alignments,word_counts)

        # reset the memory
        self.phrase_equal = None
        self.phrase_equal = defaultdict(lambda: []*3)


    def _update_moses(self, src, tgt, word_alignments, word_counts):
        ''' Update following variables: word counts e2f, f2e, phrase count e, f
        '''
        # phrase count
        if (src in self.moses_interface.phrase_count_e):
            self.moses_interface.phrase_count_e[src]+= word_counts[2]
        else:
            self.moses_interface.phrase_count_e[src] = word_counts[2]

        if (tgt in self.moses_interface.phrase_count_f):
            self.moses_interface.phrase_count_f[tgt]+= word_counts[2]
        else:
            self.moses_interface.phrase_count_f[tgt] = word_counts[2]

        srcphrase = src.split(b' ')
        tgtphrase = tgt.split(b' ')

        for src_id, tgt_lst in word_alignments.iteritems():
            for tgt_id in tgt_lst:
                self.moses_interface.word_pairs_e2f[srcphrase[src_id]][tgtphrase[tgt_id]] += word_counts[2]
                self.moses_interface.word_pairs_f2e[tgtphrase[tgt_id]][srcphrase[src_id]] += word_counts[2]

        return None

    def _get_features_Cohn(self,src,target,feature1,feature2):
        """from the Moses phrase table probability, get the new probability
           TODO: the phrase penalty value?
        """
        phrase_features =  [0]*4
        phrase_features[0] = feature1[2] * feature2[0]
        phrase_features[1] = feature1[3] * feature2[1]
        phrase_features[2] = feature1[0] * feature2[2]
        phrase_features[3] = feature1[1] * feature2[3]

        return phrase_features


    def _get_word_alignments(self,src,target,align1,align2):
        """from the Moses phrase table alignment info in the form "0-0 1-0",
           get the aligned word pairs / NULL alignments
        """
        phrase_ps = defaultdict(lambda: [])
        phrase_pt = defaultdict(lambda: [])
        # fill value to the phrase_align
        try:
            for pair in align1.split(b' '):
                p,s = pair.split(b'-')
                p,s = int(p),int(s)
                phrase_ps[p].append(s)
            for pair in align2.split(b' '):
                p,t = pair.split(b'-')
                p,t = int(p),int(t)
                phrase_pt[p].append(t)
        except:
            pass

        # 20150104: fix the alignment error
        phrase_st = defaultdict(lambda: []*3)
        for pvt_id, src_lst in phrase_ps.iteritems():
            if (pvt_id in phrase_pt):
                tgt_lst = phrase_pt[pvt_id]
                for src_id in src_lst:
                    for tgt_id in tgt_lst:
                        if (tgt_id not in phrase_st[src_id]):
                            phrase_st[src_id].append(tgt_id)
        return phrase_st


    def _get_word_counts(self,src,target,count1,count2):
        """from the Moses phrase table word count info in the form "1000 10 10",
           get the counts for src, tgt
           the word count is: target - src - both
        """
        word_count = [0]*3

        if (len(count1) > 2):
            word_count[2] = self.computed(long(float(count1[2])),long(float(count2[2])))
        return word_count

    def _write_phrasetable(self,model1,model2,output_object,inverted=False):
        """Incrementally load phrase tables, calculate score for increment and write it to output_object"""
        # define which information we need to store from the phrase table
        # possible flags: 'all', 'target', 'source' and 'pairs'
        # interpolated models without re-normalization only need 'pairs', otherwise 'all' is the correct choice
        store_flag = 'all'
        if self.mode == 'interpolate' and not self.flags['normalized']:
            store_flag = 'pairs'

        sys.stderr.write('Incrementally loading and processing phrase tables...')
        # Start process phrase table
        self.phrase_equal = defaultdict(lambda: []*3)
        self._phrasetable_traversal(model1=model1, model2=model2, prev_line1=None, prev_line2=None, deci=0, output_object=output_object,iteration=0)

        sys.stderr.write("done")


# GLOBAL DEF
def handle_file(filename,action,fileobj=None,mode='r'):
    """support reading/writing either from/to file, stdout or gzipped file"""

    if action == 'open':

        if mode == 'r':
            mode = 'rb'
        elif mode == 'w':
            mode = 'wb'

        if mode == 'rb' and not filename == '-' and not os.path.exists(filename):
            if os.path.exists(filename+'.gz'):
                filename = filename+'.gz'
            else:
                sys.stderr.write('Error: unable to open file. ' + filename + ' - aborting.\n')

                if 'counts' in filename and os.path.exists(os.path.dirname(filename)):
                    sys.stderr.write('For a weighted counts combination, we need statistics that Moses doesn\'t write to disk by default.\n')
                    sys.stderr.write('Repeat step 4 of Moses training for all models with the option -write-lexical-counts.\n')

                exit(1)

        if filename.endswith('.gz'):
            fileobj = gzip.open(filename,mode)

        elif filename == '-' and mode == 'wb':
            fileobj = sys.stdout

        else:
                fileobj = open(filename,mode)

        return fileobj

    elif action == 'close' and filename != '-':
        fileobj.close()


def sort_file(filename,tempdir=None):
    """Sort a file and return temporary file"""

    cmd = ['sort', filename]
    env = {}
    env['LC_ALL'] = 'C'
    if tempdir:
        cmd.extend(['-T',tempdir])

    outfile = NamedTemporaryFile(delete=False,dir=tempdir)
    sys.stderr.write('LC_ALL=C ' + ' '.join(cmd) + ' > ' + outfile.name + '\n')
    p = Popen(cmd,env=env,stdout=outfile.file)
    p.wait()

    outfile.seek(0)

    return outfile


def dot_product(a,b):
    """calculate dot product from two lists"""

    # optimized for PyPy (much faster than enumerate/map)
    s = 0
    i = 0
    for x in a:
        s += x * b[i]
        i += 1

    return s

def get_minimum_counts(count1, count2):
    ''' get the mimimum occurrences between two occurrences
    '''
    return min(count1,count2)

def get_maximum_counts(count1, count2):
    ''' get the maximum occurrences between two occurrences
    '''
    return max(count1,count2)

def get_arithmetic_mean(count1, count2):
    ''' get arithmetic mean between two numbers
    '''
    return (count1+count2)/2
def get_geometric_mean(count1, count2):
    ''' get the geometric mean between two numbers
    '''
    return sqrt(count1*count2)



def _load_line(line):
    if (not line):
        return None
    ''' This function convert a string into an array of string and probability
        src ||| tgt ||| s|t s|t t|s t|s ||| align ||| countt counts countst ||| |||
    '''
    #print "line : ", line
    line = line.rstrip().split(b'|||')
    if line[-1].endswith(b' |||'):
        line[-1] = line[-1][:-4]
        line.append(b'')

    # remove the blank space
    line[0] = line[0].strip()
    line[1] = line[1].strip()

    # break the probability
    line[2]  = [float(i) for i in line[2].strip().split(b' ')]

    # break the alignment
    phrase_align = defaultdict(lambda: []*3)
    for pair in line[3].strip().split(b' '):
        try:
            s,t = pair.split(b'-')
            s,t = int(s),int(t)
            phrase_align[s].append(t)
        except:
            #print "Infeasible pair ", pair
                pass
    line[3] = phrase_align
    # break the count
    # sometimes, the count is too big, save it as long
    line[4] = [long(float(i)) for i in line[4].strip().split(b' ')]

    return line

def _write_phrasetable_file(line):
    '''
    write the phrase table line
    '''
    # convert data to appropriate format
    # probability
    src,tgt,features,alignment,word_counts = line[:5]
    features = b' '.join([b'%.6g' %(f) for f in features])

    alignments = []
    for src_id,tgt_id_list in alignment.iteritems():
        for tgt_id in sorted(tgt_id_list):
            alignments.append(str(src_id) + '-' + str(tgt_id))
    extra_space = b''
    if(len(alignments)):
        extra_space = b' '
    alignments = b' '.join(str(x) for x in alignments)

    word_counts = b' '.join([b'%.6g' %(f) for f in word_counts])

    outline = b"%s ||| %s ||| %s ||| %s%s||| %s ||| |||\n" %(src,tgt,features,alignments,extra_space,word_counts)
    return outline


if __name__ == "__main__":

    if len(sys.argv) < 2:
        sys.stderr.write("no command specified. use option -h for usage instructions\n")

    elif sys.argv[1] == "test":
        test()

    else:
        args = parse_command_line()
        #initialize
        combiner = Triangulate_TMs(weights=args.weights,
                               model1=args.srcpvt,
                               model2=args.pvttgt,
                               mode=args.mode,
                               output_file=os.path.normpath('/'.join([args.tempdir2, 'phrase-table'])),
                               inverted=args.inverted,
                               action=args.action,
                               computed=args.computation,
                               reference_file=args.reference,
                               output_lexical=args.output_lexical,
                               lowmem=args.lowmem,
                               normalized=args.normalized,
                               recompute_lexweights=args.recompute_lexweights,
                               tempdir=args.tempdir2,
                               number_of_features=args.number_of_features,
                               i_e2f=args.i_e2f,
                               i_e2f_lex=args.i_e2f_lex,
                               i_f2e=args.i_f2e,
                               i_f2e_lex=args.i_f2e_lex,
                               write_phrase_penalty=args.write_phrase_penalty)

        # write everything to a file
        combiner.combine_standard()
        # sort the file
        tmpfile = sort_file(combiner.output_file,tempdir=args.tempdir2)
        #os.remove(combiner.output_file)
        # combine the new file
        merger = Merge_TM(model=tmpfile,
                          output_file=args.output,
                          mode=combiner.mode,
                          action=args.action,
                          moses_interface=combiner.moses_interface)
        merger._combine_TM()
