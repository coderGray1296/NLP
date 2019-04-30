# -*- coding: utf-8 -*-
import sys
import re
import codecs
import glob
from collections import defaultdict
from process_hit import ProcessHit
from process_conll import ProcessConll

class ParseHit:
    def __init__(self, fid, adjudicate, unique_id, NEs_given=True, gold=False):
        self.gold = gold
        self.NEs_given = NEs_given
        self.check_controls = defaultdict(lambda: defaultdict(list))
        self.adjudicate = adjudicate
        self.UniqueID = unique_id
        self.html_escape_table = {"&AMP;":"&", "&amp;":"&", "&QUOT;":'"', "&quot;":'"', "&APOS;":"'", "&apos;":"'", "&GT;":">", "&gt;":">", "&LT;":"<", "&lt;":"<", "&#44;":",", "<BR />":"\n", "<br />":"\n", "<BR/>":"\n", "<br/>":"\n"}
        self.count_hash = {}
        self.aggregated_answers = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.consensus_hash = defaultdict(lambda: defaultdict(str))
        self.possible_controls = defaultdict(lambda: defaultdict(str))
        self.new_controls = {}
        hit_obj = ProcessHit(fid, NEs_given=NEs_given)
        self.answers = hit_obj.get_answers()
        self.aggregate_answers()
        self.get_consensus()
        
    def aggregate_answers(self):
        for hit_id  in self.answers:
            for worker_id in self.answers[hit_id]:
                for (tweet_id, entity, tweet_before, tweet_after) in self.answers[hit_id][worker_id]:
                    if not self.NEs_given:
                        vote = self.answers[hit_id][worker_id][(tweet_id, entity, tweet_before, tweet_after)]
                    else:
                        vote = self.answers[hit_id][worker_id][(tweet_id, entity, tweet_before, tweet_after)]
                    if re.search("control", tweet_id):
                        self.check_controls[tweet_id][vote] += [worker_id]
                        split_tweet_id = tweet_id.split("_")
                        tweet_id = split_tweet_id[0]
                        #vote = split_tweet_id[-1]
                        #continue
                    self.aggregated_answers[tweet_id][(entity, tweet_before, tweet_after)][vote] += [worker_id]
    
    def get_consensus(self):
        # Consensus:  At least 2 out of 3 people agree (for overall)
        for tweet_id in self.aggregated_answers:
            #if re.search("_control", tweet_id):
            #    continue
            for (entity, tweet_before, tweet_after) in self.aggregated_answers[tweet_id]:
                ordered_list = []
                for (vote, worker_ids) in self.aggregated_answers[tweet_id][(entity, tweet_before, tweet_after)].iteritems():
                    count = len(worker_ids)
                    ordered_list += [(count, vote)]
                ordered_list.sort()
                ordered_list.reverse()
                (top_count, top_vote) = ordered_list[0]
                if top_count >= 2 or self.gold:
                    self.consensus_hash[tweet_id][(entity, tweet_before, tweet_after)] = top_vote
                else:
                    self.consensus_hash[tweet_id][(entity, tweet_before, tweet_after)] = None
                if top_count >= 3:
                    self.possible_controls[tweet_id][(entity, tweet_before, tweet_after)] = top_vote

    def decide_whether_to_print(self, tweet_conll_input, sentiment_based):
        NE_good = False
        sentiment_good = False
        for word in tweet_conll_input:
            NE = tweet_conll_input[word][1].strip()
            sentiment = tweet_conll_input[word][-1].strip()
            if NE == "B-PERSON" or NE == "B-ORGANIZATION":
                NE_good = True
                if sentiment_based and sentiment == "_":
                    return False
        return NE_good            
        
    def print_to_conll(self, conll_input, out_file, print_all, sentiment_based):
        start = True
        for tweet_id in sorted(conll_input):
            tweet_conll_input = conll_input[tweet_id]
            #if print_all or not self.NEs_given:
            #    to_print = True
            #else:
            #    to_print = self.decide_whether_to_print(tweet_conll_input, sentiment_based)
            # We assume that the remaining relevant material is stored in consensus_hash
            # For everything the Turkers saw, get the subset that had consensus;
            # bits without consensus (marked None) are removed if requested.
            for (html_entity, html_tweet_before, html_tweet_after) in self.consensus_hash[tweet_id]:
                vote = self.consensus_hash[tweet_id][(html_entity, html_tweet_before, html_tweet_after)]
                if vote == None:
                    if not self.NEs_given:
                        NE_label = 'O'
                        sentiment = "_"
                    else:
                        sentiment = "_"
                        #if sentiment_based:
                        #to_print = False
                        # Don't print.
                        #break
                else:
                    if not self.NEs_given:
                        sentiment = vote[0]
                        NE_label = vote[1]
                    else:
                        sentiment = vote
                #to_print = True
                # Get the NE span from the conll data.
                span = self.get_entity_span(tweet_id, html_entity, html_tweet_before, html_tweet_after)
                start_span = span[0]
                # If we have an answer to this and are changing it, alert.
                if tweet_conll_input[start_span][-1] != "_" and sentiment != "_" and (sentiment != tweet_conll_input[start_span][-1]):
                    sys.stderr.write("Warning: " + tweet_id + " Changing " + tweet_conll_input[start_span][-1] + " to " + sentiment + "\n")
                    sys.stderr.write("\t".join(tweet_conll_input[start_span]) + "\n")
                    #sys.exit()
                tweet_conll_input[start_span][-1] = sentiment
                if not self.NEs_given:
                    if NE_label == "O" or NE_label == "_":
                        tweet_conll_input[start_span][1] = "O"
                        for next_word in span[1:]:
                            tweet_conll_input[next_word][1] = "O"
                    else:
                        tweet_conll_input[start_span][1] = "B-" + NE_label
                        for next_word in span[1:]:
                            tweet_conll_input[next_word][1] = "I-" + NE_label
                    #print "adding ", NE, "to", tweet_conll_input[start_span]
                # If we're adjudicating, mark all the votes.
                if self.adjudicate:
                    for (sentiment, worker_ids) in sorted(self.aggregated_answers[tweet_id][(html_entity, html_tweet_before, html_tweet_after)].iteritems()):
                        tweet_conll_input[start_span] += [sentiment + ":" + str(len(worker_ids))]
                # Find new controls.
                if tweet_id in self.possible_controls:
                    if (html_entity, html_tweet_before, html_tweet_after) in self.possible_controls[tweet_id]:
                        if self.has_one_polarity(tweet_conll_input):
                            self.new_controls[tweet_id] = tweet_conll_input
            if print_all:
                to_print = True
            else:
                to_print = self.decide_whether_to_print(tweet_conll_input, sentiment_based)
            if to_print:
                if start:
                    out_file.write(self.UniqueID + " " + str(tweet_id) + "\n")
                    start = False
                else:
                    out_file.write("\n" + self.UniqueID + " " + str(tweet_id) + "\n")
                self.print_out(tweet_conll_input, out_file)
        #sys.stderr.write("Controls have (but we don't): " + " ".join(missing) + "\n")
    
    def has_one_polarity(self, tweet_conll_input):
        num_polarity = 0
        for line_num in tweet_conll_input:
            split_line = tweet_conll_input[line_num]
            polarity = split_line[-1]
            if polarity != "_":
                num_polarity += 1
        if num_polarity == 1:
            return True
        return False
        
    def print_out(self, tweet_conll_input, out_file):
        for line_num in sorted(tweet_conll_input):
            features = "\t".join(tweet_conll_input[line_num][1:]).decode('utf8')
            word = tweet_conll_input[line_num][0]
            word = word.decode('utf-8')#.encode("unicode_escape")
            out = word + "\t" + features
            #out = unicode(out, 'unicode-escape')
            out_file.write(out + "\n")

    def get_entity_span(self, tweet_id, html_entity, html_tweet_before, html_tweet_after):
        start = None
        entity = html_entity
        tweet_before = html_tweet_before
        if html_tweet_before == "&nbsp;":
            tweet_before = ""
            start = 0
        for (s, sub) in self.html_escape_table.iteritems():
            tweet_before = re.sub(s, sub, tweet_before)
        if start is None:
            split_before = tweet_before.split(" ")
            start = len(split_before)
        split_entity = entity.split(" ")
        end = len(split_entity)
        span = tuple(range(start,start+end))
        return span
            
    def print_consensus(self):
        no_consensus = 0.0
        yes_consensus = 0.0
        for tweet_id in sorted(self.consensus_hash):
            for (entity, tweet_before, tweet_after) in self.consensus_hash[tweet_id]:
                #if "_control" in tweet_id:
                #    continue
                answer = self.consensus_hash[tweet_id][(entity, tweet_before, tweet_after)]
                if answer == None:
                    no_consensus += 1
                else:
                    yes_consensus += 1
                #print tweet_id, self.consensus_hash[(tweet_id, entity, tweet_before, tweet_after)]
        sys.stderr.write("Consensus: " + str(yes_consensus) + " of " + str(no_consensus + yes_consensus) + ": " + str(round(yes_consensus/float(no_consensus + yes_consensus), 4)) + "\n")
        
    def print_check_controls(self):
        print len(self.check_controls), "controls."
        for tweet_id in self.check_controls:
            control_sentiment = tweet_id.split("_")
            control_sentiment = control_sentiment[-1]
            ordered_list = []
            for (sentiment, worker_ids) in self.check_controls[tweet_id].iteritems():
                count = len(worker_ids)
                ordered_list += [(count, sentiment)]
            ordered_list.sort()
            ordered_list.reverse()
            (top_count, top_sentiment) = ordered_list[0]
            if top_sentiment != control_sentiment:
                print tweet_id, top_sentiment, control_sentiment, ordered_list

    def print_new_controls(self):
        print "Possible new controls:"
        for tweet_id in self.new_controls:
            #print tweet_id
            tweet_conll_input = self.new_controls[tweet_id]
            print self.possible_controls[tweet_id].keys()
            print "\n## Tweet " + tweet_id
            for line_num in sorted(tweet_conll_input):
                print "\t".join(tweet_conll_input[line_num])

if __name__ == "__main__":
    NEs_given = True
    if "--no-NEs" in sys.argv[2:]:
        NEs_given = False
    UniqueID = "## Tweet"
    if len(sys.argv) < 4:
        sys.stderr.write("Usage: %prog batch_file conll_file out_file\n" )
        sys.exit()
    # Batch file
    f = codecs.open(sys.argv[1], "r", 'utf-8')
    fid = f.read()
    # Hack: Windows encoding snuck in.
    fid = re.sub("\r", "", fid)
    fid = fid.split("\n")
    f.close()
    # Conll file
    g = codecs.open(sys.argv[2], "r")
    gid = g.readlines()
    g.close()
    out_file = codecs.open(sys.argv[3], "w+", 'utf-8')
    if "--gold" in sys.argv[3:]:
        gold = True
    else:
        gold = False
    if "--print-all" in sys.argv[3:]:
        print_all = True
    else:
        print_all = False
    # Do we want to remove tweets where not all NE sentiment is filled out?
    if "--sentiment-based" in sys.argv[3:]:
        sentiment_based = True
    else:
        sentiment_based = False
    adjudicate = False
    test_obj = ProcessConll(gid, UniqueID, NEs_given=NEs_given)
    tweet_to_entity = test_obj.get_tweets_to_entities()
    conll_input = test_obj.get_conll_input()
    hit_obj = ParseHit(fid, adjudicate, UniqueID, NEs_given=NEs_given, gold=gold)
    hit_obj.print_to_conll(conll_input, out_file, print_all, sentiment_based)
    #hit_obj.print_consensus()
    #hit_obj.print_check_controls()
    #hit_obj.print_new_controls()
