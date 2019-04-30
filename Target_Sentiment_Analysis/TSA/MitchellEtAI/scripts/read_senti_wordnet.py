import sys
import re

fid = sys.stdin.readlines()

def calc_sentiment(posScore, negScore):
    p = float(posScore)
    n = float(negScore)
    if p + n > .5:
        if p > n:
            return "positive"
        else:
            return "negative"
    else:
        return "neutral"

sentiment_hash = {}
for line in fid:
    if line.startswith("#"):
        continue
    else:
        split_line = line.split()
        posScore = split_line[2]
        negScore = split_line[3]
        for word in split_line[4:]:
            if re.search("[^#]+#\d+", word):
                split_word = word.split("#")
                num = split_word[1]
                word = split_word[0]
                if num == "1":
                    sentiment = calc_sentiment(posScore, negScore)
                    sentiment_hash[word] = sentiment
            else:
                break
