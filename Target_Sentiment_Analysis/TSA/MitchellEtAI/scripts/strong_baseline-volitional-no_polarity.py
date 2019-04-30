#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import re
from collections import defaultdict

fid = sys.stdin.readlines()
linear = False
if "--linear" in sys.argv[1:]:
    linear = True

def get_sentiment(prev_context, next_context):
    sentiment_hash = {"SENT:positive":0, "SENT:negative":0}
    for line in prev_context:
        split_line = line.split("\t")
        sentiment = split_line[5]
        if sentiment == "SENT:positive" or sentiment == "SENT:negative":
            sentiment_hash[sentiment] += 1
    for line in next_context:
        split_line = line.split("\t")
        sentiment = split_line[5]
        if sentiment == "SENT:positive" or sentiment == "SENT:negative":
            sentiment_hash[sentiment] += 1
    if sentiment_hash["SENT:positive"] > sentiment_hash["SENT:negative"]:
        overall_sentiment = "sentiment"
    elif sentiment_hash["SENT:negative"] > sentiment_hash["SENT:positive"]:
        overall_sentiment = "sentiment"
    else:
        overall_sentiment = "neutral"
    return overall_sentiment

def check_for_volitional(fid, y):
    orig_y = y
    next_line = fid[y]
    has_volitional = False
    while not next_line.startswith("## Tweet"):
        next_line = next_line.strip()
        if next_line == "":
            pass
        else:
            split_line = next_line.split("\t")
            label = split_line[1]
            if label == "B-PERSON" or label == "B-ORGANIZATION":
                has_volitional = True
                return (has_volitional, orig_y)
        y += 1
        try:
            next_line = fid[y]
        except IndexError:
            break
    return (has_volitional, y)

def get_predictions(fid):
    predictions = {}
    x = 0
    tweetid = -1
    need_sentiment = False
    while x < len(fid):
        line = fid[x]
        line = line.strip()
        if line.startswith("## Tweet"):
            n = 0
            # This will move us up if the tweet
            # doesn't have a volitional entity.
            # I don't think this actually happens anymore,
            # I'm screening them out earlier.
            (has_volitional, x) = check_for_volitional(fid, x+1)
            while not has_volitional:
                if x >= len(fid):
                    return predictions
                (has_volitional, x) = check_for_volitional(fid, x+1)
            tweetid += 1
            predictions[tweetid] = {}
            continue
        elif line == "":
            x += 1
            continue
        else:
            split_line = line.split("\t")
        word = split_line[0]
        if re.search("[A-ZÁÉÍÓÚËÜ]", word[0]):
            # It's the most common NE.
            #print "Guessing person for ", word
            if linear:
                pass
            else:
                predictions[tweetid]["W" + str(n)] = "B_VOLITIONAL"
            need_sentiment = True
        else:
            predictions[tweetid]["W" + str(n)] = "O"
            need_sentiment = False
        if n < 3:
            prev_context = fid[x-n:x]
        else:
            prev_context = fid[x-3:x]
        next_context = fid[x+1:x+4]
        o = 1
        for next_line in next_context:
            if next_line.strip() == "":
                next_context = fid[x+1:x+o-1]
                break
            o += 1
        sentiment = get_sentiment(prev_context, next_context)
        if need_sentiment:
            if linear:
                predictions[tweetid]["W" + str(n)] = "B" + sentiment
            else:
                predictions[tweetid]["S" + str(n)] = sentiment
        else:
            if linear:
                pass
            else:
                predictions[tweetid]["S" + str(n)] = "neutral"
        n += 1
        x += 1
    return predictions

predictions = get_predictions(fid)
for tweetid in sorted(predictions):
    print "//example " + str(tweetid)
    print "example:"
    for var in sorted(predictions[tweetid]):
        print var + "o=" + predictions[tweetid][var] 
