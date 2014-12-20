#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  This class implement a method to automatically extract the pair of language in one directory
#  it take the input as a "src" language and a "tgt" language and a "directory" of corpora
#  it outputs all the phrase pair necessary: direct parallel corpora, a list of all possible way to do triangulation and a list of monolingual corpora of the target language

from __future__ import division, unicode_literals
import sys
import os
import gzip
import argparse
import copy
import re
from math import log, exp
from collections import defaultdict
from operator import mul
from tempfile import NamedTemporaryFile
from tmcombine import Moses, Moses_Alignment, to_list

try:
    from itertools import izip
except:
    izip = zip

def parse_command_line():
    parser = argparse.ArgumentParser(description='Find the pair of data given in a directory, all the options will be implemented later')

    group1 = parser.add_argument_group('Main options')

    group1.add_argument('-s', metavar='DIRECTORY', dest='source',
                    help='source language that you want to translate from')
    group1.add_argument('-t', metavar='DIRECTORY', dest='target',
                    help='target language that you want to translate to')
    group1.add_argument('-c', metavar='DIRECTORY', dest='corporadir',
                    help='the directory which contains all corpora')
    group1.add_argument('-o', metavar='DIRECTORY', dest='output_file',
                    help='the output file')

    return parser.parse_args()


class Decode_Corpora():
    """This class handles the process of looking for a specific
       it also check the correctness of the bilingual corpus which they have the same line number or not
    """

    def __init__(self,source=None,
                      target=None,
                      corporadir=None,
                      output_file=None):

        ''' Initialize the value, list all the folder in the directory
        '''

        self.source = source
        self.target = target
        self.cordir = corporadir
        self.output_file = output_file

        self._load_config()
        self._browse_dir()

    def _load_config(self, config_file='config.properties'):
        '''
        load the configuration of the whole folder in a file
        '''
        return None

    def _browse_dir(self):
        ''' browse the directory and load a hash table out of those dir
        '''
#TODO: There might be a nicer way to concatenate directory
        print "Browsing directory"
        self.structure = defaultdict()
        self.structure[0] = os.path.normpath('/'.join([os.getcwd(), self.cordir]))
        print self.structure[0]
        # load all directories inside
        onlydirs = [d for d in os.listdir(self.structure[0]) if os.path.isdir('/'.join([self.structure[0], d]))]
        print onlydirs
        self.structure[1] = defaultdict(lambda: [])
        # loop through 1st layer directory
        for dir_f in onlydirs:
            if (len(dir_f)%2):
                print "Ignore folder ", dir_f, " because of incorrect format"
                continue
            dir_type = int(len(dir_f)/2)
            i = 0
            languages = []
            self.structure[1][dir_f].append(languages)
            while(i+2 <= len(dir_f)):
                dir_f2 = dir_f[i:(i+2)]
                print dir_f2
                languages.append(dir_f2)
                i+=2
            dir_abs = os.path.normpath('/'.join([self.structure[0], dir_f]))
            onlyfiles = [f for f in os.listdir(dir_abs) if (os.path.isfile('/'.join([dir_abs, f])) and f[:1].isalpha())]
            print onlyfiles
            # loop through the 2nd layer of directory
            # not used anymore
            if (dir_type == 1):
                # monolingual folder
                for onlyfile in onlyfiles:
                    if (onlyfile.endswith(dir_f)):
                        pass
                        #self.structure[1][dir_f].append(only_file)
            #
            if (len(onlyfiles) < dir_type):
                print "WRONG SIZE", len(onlyfiles), dir_type+1
            fileslist = sorted(onlyfiles)
            for file_id in range(len(onlyfiles)-dir_type+1):
                print len(fileslist[file_id:file_id+dir_type])
                if (self._check_fileslist(fileslist[file_id:file_id+dir_type], languages, dir_type)):
                    print "match: ", file_id
                    self.structure[1][dir_f].append([os.path.normpath('/'.join([dir_abs,f])) for f in fileslist[file_id:file_id+dir_type]])
                    file_id += dir_type
        print self.structure

    def _check_fileslist(self, fileslist, languageslist, dir_type):
        ''' get the list of files which match the languagelist
        '''
        if (len(fileslist) != len(languageslist)):
            print "mistake of matching, you dumb ass"
            return False
        for file_f,lang_f in zip(fileslist,languageslist):
            if (not file_f.endswith("."+lang_f)):
                return False
        return True

    def _check_file(self, file_f, lang_f):
        ''' check if the file fit the description
        '''
        if (file_f.endswith("."+lang_f)):
                return True
        return False

    def _sanity_checks(self):
        """check if the corpora dir is clean, which mean:
           + the folder has to be in the alphabet order
           + in a bilingual directory, there must be an even number of file, in which each pair only different by the language in the folder
        """
        return True

    def _monolingual_find(self, target=None):
        """ find all the monolingual corpora for the language model
            return a list of files
            maybe another list of parallel corpora which could be used as monolingual corpora
        """
        mono, multi = [],[]
        for lang,dir_f in self.structure[1].iteritems():
            for fileslist in dir_f[1:]:
                for file_f in fileslist:
                    if (self._check_file(file_f,target)):
                        if (len(dir_f[0]) == 1):
                            mono.append(file_f)
                        else:
                            multi.append(file_f)

        print "Mono", mono
        print "multi", multi


        return None

    def _bilingual_find(self):
        """ find all the bilingual corpora for the language model
            return a list of paired files
            (cs1,uk1), (cs2,uk2)
        """
        return None

    def _triangulation_find(self):
        """ find all the bilingual corpora for the language model
            return a dictionary
            dict -> {cs-en-uk}->[0] = [[cs1,en1],[cs2,en2]]
                              ->[1] = [[en,uk],[en1,uk1]]
            TODO: Think about the reverse pivot-* config
        """
        return None



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



def dot_product(a,b):
    """calculate dot product from two lists"""

    # optimized for PyPy (much faster than enumerate/map)
    s = 0
    i = 0
    for x in a:
        s += x * b[i]
        i += 1

    return s



if __name__ == "__main__":

    if len(sys.argv) < 2:
        sys.stderr.write("no command specified. use option -h for usage instructions\n")

    elif sys.argv[1] == "test":
        test()

    else:
        args = parse_command_line()
        #initialize
        print "OK, Let's play!"
        dc = Decode_Corpora(source=args.source,
                               target=args.target,
                               corporadir=args.corporadir,
                               output_file=args.output_file)
        dc._monolingual_find("en")
