import sys
import re
from optparse import OptionParser

usage = "Need to fill this out."

parser = OptionParser(usage=usage)
parser.add_option("-u", "--unique-id", type="string", help="Unique identifier for each sentence.", default="## Tweet", dest="UniqueID")
(options, args) = parser.parse_args()
UniqueID = options.UniqueID

annot_file = open(sys.argv[1], "r").readlines()
clusters5_file = open(sys.argv[2], "r").readlines()
jerboa_file = open(sys.argv[3], "r").readlines()
clusters3_file = open(sys.argv[4], "r").readlines()
sentiment_file = open(sys.argv[5], "r").readlines()

if len(annot_file) == len(jerboa_file) == len(sentiment_file) == len(clusters5_file) == len(clusters3_file):
    pass
else:
    sys.stderr.write("Error:  Features files don't have same number of lines.\n")
    sys.stderr.write("Annotation file: " + str(len(annot_file)) + " Clusters files: " + str(len(clusters5_file)) + " "+ str(len(clusters3_file)) + " Jerboa file:" + str(len(jerboa_file)) + " Sentiment file:" + str(len(sentiment_file)) + "\n")
    sys.exit()

x = 0
while x < len(annot_file):
    line = annot_file[x].strip()
    if line.startswith(UniqueID):
        print line
    elif line == "":
        print line
    else:
        split_line = line.split("\t")
        cluster5 = clusters5_file[x].strip()
        cluster3 = clusters3_file[x].strip()
        jerb = jerboa_file[x].strip()
        if jerb == "null":
            jerb = "NONE"
        rada_ther_sent = sentiment_file[x].strip()
        sent_split = rada_ther_sent.split()
        rada_sent = sent_split[0]
        ther_sent = sent_split[1]
        print split_line[0] + "\t" + split_line[1] + "\t" + cluster5 + "\t" + jerb + "\t" + cluster3 + "\t" + rada_sent + "\t" + ther_sent + "\t" + split_line[-1]
    x += 1
