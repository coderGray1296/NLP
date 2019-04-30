import sys
import re

class ProcessConll:
    def __init__(self, fid, unique_id="## Tweet", controls=False, NEs_given=True):
        self.NEs_given = NEs_given
        self.tag_hash = {}
        self.polarity_hash = {}
        self.conll_input = {}
        self.sent_to_controls = {"positive":[], "negative":[], "neutral":[]}
        self.fid = fid
        self.controls = controls
        self.unique_id = unique_id
        self.out_tweet_hash = {}
        self.out_tweet_hash_with_sentiment = {}
        self.tweets_to_entities = {}
        self.input_to_hash()
    
    def get_sent_to_controls(self):
        return self.sent_to_controls
        
    def get_hash(self):
        return self.out_tweet_hash
        
    def get_sentimental_hash(self):
        return self.out_tweet_hash_with_sentiment 
                
    def get_tags(self):
        return self.tag_hash
    
    def get_polarities(self):
        return self.polarity_hash
        
    def get_conll_input(self):
        return self.conll_input
        
    def get_tweets_to_entities(self):
        for tweet_num in self.out_tweet_hash:
            (tweet, NEs) = self.out_tweet_hash[tweet_num]
            split_tweet = tweet.split(" ")
            for n in NEs:
                span = tuple(sorted(NEs[n].keys()))
                ent = ""
                for word_idx in span[:-1]:
                    ent += split_tweet[word_idx] + " "
                ent += split_tweet[span[-1]]
                self.tweets_to_entities[(tweet_num, span)] = ent
        return self.tweets_to_entities
    
    def get_NEs_from_controls(self, tweet_num, word, NE, sentiment, NEs, n, o):
        # For controls, we only want things we have
        # known sentiment for.
        if (NE == "B-ORGANIZATION" or NE == "B-PERSON") and sentiment != "_":
            n += 1
            NEs[n] = {o:word}
            self.sent_to_controls[sentiment] += [(tweet_num, n)]
        elif NE == "I-ORGANIZATION" or NE == "I-PERSON":
            # If it's not in the dictionary,
            # it doesn't have sentiment associated to it.
            try:
                if o - 1 in NEs[n]:
                    NEs[n][o] = word
            except KeyError:
                pass
        return (NEs, n)
                        
    def get_NEs_from_test(self, tweet_num, word, NE, sentiment, NEs, n, o):
        if not self.NEs_given:
            return ({}, n)
        # For non-controls, we want things we don't
        # have known sentiment for.
        if (NE == "B-ORGANIZATION" or NE == "B-PERSON" or NE == "B-VOLITIONAL") and sentiment == "_":
            n += 1
            NEs[n] = {o:word}
        elif NE == "I-ORGANIZATION" or NE == "I-PERSON" or NE == "I":
            # If it's not in the dictionary,
            # it has sentiment associated to it.
            try:
                if o - 1 in NEs[n]:
                    NEs[n][o] = word
            except KeyError:
                pass
        return (NEs, n)

    def input_to_hash(self):
        o = 0
        tweet_num = "0"
        for line in self.fid:
            strip_line = line.strip()
            split_line = strip_line.split()
            if strip_line == "":
                continue
            elif re.search(self.unique_id, strip_line):
                # If we're just starting, nothing to store.
                if tweet_num != "0":
                    # If we have relevant NEs, include.
                    if NEs != {}:
                        self.out_tweet_hash[tweet_num] = (tweet, NEs)
                        NEs_sent = (NEs, NEs_sentiment)
                        self.out_tweet_hash_with_sentiment[tweet_num] = (tweet, NEs_sent)
                n = 0
                o = 0
                last_n = 0
                tweet_num = split_line[-1]
                tweet = ""
                NEs = {}
                NEs_sentiment = {}
                self.tag_hash[tweet_num] = {}
                self.polarity_hash[tweet_num] = {}
                self.conll_input[tweet_num] = {}
                continue
            else:
                #print split_line
                word = split_line[0]#.encode('utf8')
                #try:
                #    word = unicode(split_line[0], "unicode_escape")#.decode("utf-8")
                #except UnicodeDecodeError:
                #    word = unicode(split_line[0]).decode("utf-8")
                if len(split_line) == 2:
                    sentiment = "_"
                elif len(split_line) > 2:
                    sentiment = split_line[-1]
                else:
                    sys.stderr.write("Sentiment problem in file.\n")
                    sys.stderr.write("Line: " + line + " " + "Tweet num: " + tweet_num + "\n")
                    sys.exit()
                NE = split_line[1]
                if self.controls:
                    last_n = n
                    (NEs, n) = self.get_NEs_from_controls(tweet_num, word, NE, sentiment, NEs, n, o)
                    if n > last_n:
                        NEs_sentiment[n] = sentiment
                else:
                    (NEs, n) = self.get_NEs_from_test(tweet_num, word, NE, sentiment, NEs, n, o)
                if tweet == "":
                    tweet += word
                else:
                    tweet += " " + word
                self.tag_hash[tweet_num][o] = NE
                self.polarity_hash[tweet_num][o] = sentiment
                self.conll_input[tweet_num][o] = [word] + split_line[1:]
                o += 1
        self.out_tweet_hash[tweet_num] = (tweet, NEs)
        NEs_sent = (NEs, NEs_sentiment)
        self.out_tweet_hash_with_sentiment[tweet_num] = (tweet, NEs_sent)
