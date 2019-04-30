import sys
import re
from collections import defaultdict
from optparse import OptionParser

global UniqueID

words = sys.stdin.readlines()

usage = "usage: cat words | %prog brown_clusters k [options]"
parser = OptionParser(usage=usage)
parser.add_option("-u", "--unique-id", type="string", help="Unique identifier for each sentence.", default="## Tweet", dest="UniqueID")
(options, args) = parser.parse_args()
UniqueID = options.UniqueID


b = open(sys.argv[1], "r")
brown_clusters = b.readlines()
b.close()
k = int(sys.argv[2])

def read_clusters(brown_clusters):
    cluster_hash = {}
    for line in brown_clusters:
        line = line.strip()
        split_line = line.split("\t")
        cluster = split_line[0]
        word = split_line[1]
        cut_cluster = cluster[:k]
        cluster_hash[word] = cut_cluster
    return cluster_hash

def words_to_clusters(words, cluster_hash):
    global UniqueID
    line_num = 0
    for word in words:
        line_num += 1
        word = word.strip()
        if word.startswith(UniqueID):
            print word
            continue
        elif word == "":
            print
        else:
            split_line = word.split()
            if len(split_line) > 1:
                sys.stderr.write("\n\nPossible error on line + " + str(line_num) + ":  Expected a single word, but found " + word + ".\n")
            word = split_line[0]
            try:
                cluster = cluster_hash[word]
            except KeyError:
                cluster = "NONE"
            print cluster


cluster_hash = read_clusters(brown_clusters)
words_to_clusters(words, cluster_hash)
