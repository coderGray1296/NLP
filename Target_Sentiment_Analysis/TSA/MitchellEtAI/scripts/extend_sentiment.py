import sys

"""
## Tweet 1000
How O   _
can O   _
someone O   _
so  O   _
incompetent O   _
like    O   _
Maxine  B-PERSON    negative
Waters  I-PERSON    _
stay    O   _
in  O   _
office  O   _
for O   _
    over    O   _
    20  B-TIME  _
    years   I-TIME  _
    ?   O   _
    """

fid = sys.stdin.readlines()

sentiment = ("positive", "negative", "neutral")
last_sentiment = "_"
for line in fid:
    line = line.strip()
    split_line = line.split("\t")
    if split_line[-1] != "_":
        last_sentiment = split_line[-1]
        extend_sentiment = True
    elif extend_sentiment:
        if split_line[1].startswith("I-"):
            line = "\t".join(split_line[:-1]) + "\t" + last_sentiment
        else:
            extend_sentiment = False
    print line

