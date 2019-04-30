import sys
import re
from sentiment import Sentiment
from optparse import OptionParser

words = sys.stdin.readlines()

usage = "Need to fill this out."
parser = OptionParser(usage=usage)
parser.add_option("-u", "--unique-id", type="string", help="Unique identifier for each sentence.", default="## Tweet", dest="UniqueID")
parser.add_option("-l", "--language", type="string", help="Sentiment language.", default="es", dest="langa")
(options, args) = parser.parse_args()
UniqueID = options.UniqueID

langa = options.langa
s = Sentiment(langa)

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
        #print word
        split_line = word.split()
        if len(split_line) > 1:
            sys.stderr.write("\n\nPossible error on line + " + str(line_num) + ":  Expected a single word, but found " + word + ".\n")
        word = split_line[0]
        previous_polarity = s.get_sentiment_polarity(word)
        ther_sentiment_polarity = s.get_ther_sentiment_polarity(word)
        print previous_polarity + "\t" + ther_sentiment_polarity
