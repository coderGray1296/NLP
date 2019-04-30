import sys
from collections import defaultdict
from optparse import OptionParser

usage = "Need to fill this out."

parser = OptionParser(usage=usage)
parser.add_option("-u", "--unique-id", type="string", help="Unique identifier for each sentence.", default="## Tweet", dest="UniqueID")
(options, args) = parser.parse_args()
UniqueID = options.UniqueID

fid = sys.stdin.readlines()
col_hash = defaultdict(lambda: defaultdict(int))
for line in fid:
    line = line.strip()
    #print line
    if line.startswith(UniqueID):
        continue
    split_line = line.split("\t")
    for i in range(len(split_line)):
        col_hash[i][split_line[i]] += 1
        #if i == 4 and split_line[i] == "_":
        #    print line
        #    sys.exit()

print "** Sanity Check **"
print "Here is what I see in the columns:"
print "First column, NE tags:"
print col_hash[1]
print "Second column, Brown clusters cut 5:"
print col_hash[2]
print "Third column, Jerboa features:"
print col_hash[3]
print "Fourth column, Brown clusters cut 3:"
print col_hash[4]
print "Fifth column, Rada sentiment lexicon:"
print col_hash[5]
print "Sixth column, Theresa sentiment lexicon:"
print col_hash[6]
print "Seventh column, the labels we are predicting:"
print col_hash[7]
