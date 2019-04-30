import sys
import codecs
from random import shuffle

in_file = codecs.open(sys.argv[1], "r", "utf-8")
out_file = codecs.open(sys.argv[2], "w+", "utf-8")
read_file = in_file.readlines()
in_file.close()
num = int(sys.argv[3])
in_hash = {}
unique_id = "## Tweet"
tweet_id = None

def print_tweet(tweet_id_tmp, in_hash_tmp, out_file):
    out_file.write(tweet_id_tmp + "\n")
    for n in sorted(in_hash_tmp[tweet_id_tmp]):
        out_file.write(in_hash_tmp[tweet_id_tmp][n] + "\n")

for line in read_file:
    line = line.strip()
    if line.startswith(unique_id):
        in_hash[line] = {}
        tweet_id = line
        n = 0
    else:
        in_hash[tweet_id][n] = line
        n += 1

tweet_ids = in_hash.keys()
shuffle(tweet_ids)

i = 0
while i < num:
    tweet_id = tweet_ids.pop()
    print_tweet(tweet_id, in_hash, out_file)
    i += 1
