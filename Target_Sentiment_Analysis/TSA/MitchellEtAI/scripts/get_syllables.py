#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Script to approximate syllable structure.
Defines onsets, nuclei, and codas by using 
patterns seen at word boundaries. """

import sys
import re
import codecs


# Reads in a brown clusters file.
#in_file = codecs.open(sys.argv[1], "r+", "utf-8")
in_words = sys.stdin.readlines()
#in_file.close()
out_file = codecs.open(sys.argv[1], "w+", "utf-8")
onset_count_hash = {}
coda_count_hash = {}
#word_start_re = re.compile([aeiouáéíóúüy])
for line in in_words:
    line = line.strip()
    if line == "":
        continue
    split_line = line.split()
    #try:
    #    word = unicode(split_line[1], 'unicode-escape')
    #except UnicodeDecodeError:
    #    sys.stderr.write("Could not decode " + word + "\n")
    #    word = unicode(word)
    #    sys.stderr.write("Changing to: " + word + "\n")
    word = unicode(split_line[1].decode('utf-8'))
    word = word.lower()
    word_start = re.findall(u"^([^aeiouáéíóúüy/\-'\"\.@123456789\+\?]+)[aeiouáéíóúüy]", word, re.UNICODE)
    if word_start == []:
        continue
    try:
        onset_count_hash[word_start[0]] += 1
    except KeyError:
        onset_count_hash[word_start[0]] = 1
    word_end = re.findall(u"[aeiouáéíóúüy]([^\?aeiouáéíóúüy/\-'\"\.@123456789\+\?]+)$", word, re.UNICODE)
    if word_end == []:
        continue
    try:
        coda_count_hash[word_end[0]] += 1
    except KeyError:
        coda_count_hash[word_end[0]] = 1

def sort_hash(count_hash):
    word_list = []
    for word_start in count_hash:
        count = count_hash[word_start]
        word_list += [(count, word_start)]
    word_list.sort()
    return word_list

onset_word_list = sort_hash(onset_count_hash)
coda_word_list = sort_hash(coda_count_hash)

word = word.encode('utf-8')
out_file.write("## onsets\n")
for (count, word) in onset_word_list:
    if count > 1000:
        out_file.write(word + "\n")
out_file.write("## vowels\n")
out_file.write(u"a\ne\ni\no\nu\ny\ná\né\ní\nó\nú\nü\n")
out_file.write("## codas\n")
for (count, word) in coda_word_list:
    if count > 1000:
        out_file.write(word + "\n")
