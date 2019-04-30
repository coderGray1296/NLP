#!/usr/bin/python
import sys
from collections import defaultdict
import re


unique_id = "## Tweet"
"""
PREDICTIONS:
//example 0
example:
W24o=O 0.9999999986955995
W25o=O 0.9999999999978046
W19o=O 0.9874464646892009
W20o=O 0.9999931927461222
W26o=O 0.9999999999290361
W17o=O 0.9999999997241236
W22o=O 0.9999999996241734
W1o=O 0.9694845164106579
W2o=O 0.9764199953351391
W27o=O 0.9999999994801562
W28o=O 0.9999961008852748
W5o=O 0.9998623827435279
W7o=O 0.9989107033006833
W8o=O 0.9999999999755582
W11o=O 0.9999999981040089

TEST:
## Tweet 1000
How O   110 NONE    110 _   _   _
can O   11111   NONE    111 _   _   _
someone O   00100   NONE    001 _   _   _
so  O   00101   NONE    001 _   _   _
incompetent O   01110   NONE    011 SENT:negative   THER_SENT_3:negative    _
like    O   00101   NONE    001 SENT_MED:positive   THER_SENT_3:positive    _
Maxine  B-PERSON    00100   NONE    001 _   _   negative
Waters  I-PERSON    11111   NONE    111 _   _   _
stay    O   1001    NONE    100 _   _   _
in  O   00111   NONE    001 _   _   _
office  O   0110    NONE    011 _   _   _
for O   00111   NONE    001 _   _   _
over    O   00101   NONE    001 _   _   _
20  B-TIME  01110   NONE    011 _   _   _
years   I-TIME  0110    NONE    011 _   _   _
?   O   000 NONE    000 _   _   _
#LAFail O   110 HASHTAG 110 _   _   _

"""

try:
    predictions = open(sys.argv[1], "r").readlines()
    test = open(sys.argv[2], "r").readlines()
except:
    sys.stderr.write("./combine_predictions_with_annot.py predictions annotated_test_file\n")
    sys.exit()

def process_predictions(predictions):
    predictions_hash = {}
    n = -1
    for line in predictions:
        line = line.strip()
        if line == "":
            continue
        if line.startswith("//"):
            n += 1
            predictions_hash[n] = {}
            RVs = True
            continue
        if line.startswith("example"):
            continue
        if RVs:
            # NESENT W16=O;
            RV_num_val = re.findall("([^\d]+)(\d+)o=(\S+)", line)
            if RV_num_val == []:
                continue
            RV_num_val = RV_num_val[0]
            RV = RV_num_val[0]
            num = int(RV_num_val[1])
            val = RV_num_val[2]
            try:
                predictions_hash[n][num][RV] = val
            except KeyError:
                predictions_hash[n][num] = {RV:val}
    #print predictions_hash
    #sys.exit()
    return predictions_hash

def add_to_test(test, predictions_hash):
    n = -1
    for line in test:
        line = line.strip()
        if line == "":
            print line
            continue
        if line.startswith(unique_id):
            n += 1
            num = 0
            print line
            continue
        else:
            predictions = sorted(predictions_hash[n][num])
            predictions.reverse()
            print line,
            print "\t\tGuessed: ",
            for RV in predictions:
                val = predictions_hash[n][num][RV]
                print val + "\t",
            print
            num += 1

predictions_hash = process_predictions(predictions)
add_to_test(test, predictions_hash)
