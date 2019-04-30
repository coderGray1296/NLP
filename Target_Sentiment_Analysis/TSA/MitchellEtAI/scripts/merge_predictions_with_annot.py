import sys
import re

annot = open(sys.argv[1], "r").readlines()
pred = open(sys.argv[2], "r").readlines()

annot_unique_id = "##"
x = 0
tweet = ""
go = False
RT = False
if len(annot) < len(pred):
    max_len = len(annot)
else:
    max_len = len(pred)
while x < max_len:
    annot_line = annot[x]
    pred_line = pred[x]
    x += 1
    if annot_line.startswith(unique_id):
        next_line = annot[x]
        tweet += annot_line
        continue
    elif annot_line.strip() == "":
        if go:
            print tweet
        tweet = ""
        go = False
        continue
    split_annot_line = annot_line.split("\t")
    split_pred_line = pred_line.split()
    pred_NE = split_pred_line[1]
    if pred_NE[0] == "B":
        NE = "B-VOLITIONAL"
        sentiment = pred_NE[1:]
        if sentiment != "neutral" and not RT:
            go = True
    elif pred_NE[0] == "I":
        NE = "I"
    else:
        NE = pred_NE
        sentiment = "_"
    tweet += split_annot_line[0] + "\t" + NE + "\t" + "\t".join(split_annot_line[2:-1]) + "\t" + sentiment + "\n"
if go:
    print tweet
