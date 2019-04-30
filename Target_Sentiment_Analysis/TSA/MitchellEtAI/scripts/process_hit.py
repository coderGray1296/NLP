import sys
import re
import codecs
from collections import defaultdict


class ProcessHit:
    def __init__(self, fid, screening=False, NEs_given=True):
        self.NEs_given = NEs_given
        self.screening = screening
        self.fid = fid
        self.html_escape_table = {"&AMP;":"&", "&amp;":"&", "&QUOT;":'"', "&quot;":'"', "&APOS;":"'", "&apos;":"'", "&GT;":">", "&gt;":">", "&LT;":"<", "&lt;":"<", "&#44;":",", "<BR />":"\n", "<br />":"\n", "<BR/>":"\n", "<br/>":"\n"}
        self.idx_hash = {}
        self.answers = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
        self.comments = defaultdict(lambda: defaultdict(str)) 
        self.processed_tweet_ids = defaultdict(int)
        self.overall_sentiment_hash = {}
        self.header = self.get_data(fid[0])
        # May not be answers for last confidence level, country, approve/reject
        self.expected_length = len(self.header) - 4
        self.get_idxs()
        self.read_file()
    
    def get_idxs(self):
        n = 0
        for a in self.header:
            self.idx_hash[a] = n
            n += 1
    
    def read_file(self):
        x = 1
        while x < len(self.fid):
            line = self.fid[x]
            line = line.strip()
            if line == "":
                x += 1
                continue
            split_line = self.get_data(line)
            while len(split_line) < self.expected_length:
                x += 1
                next_line = self.fid[x]
                next_split_line = self.get_data(next_line)
                # Combine new line character breaking up a string with previous string.
                split_line = split_line[:-1] + [split_line[-1] + next_split_line[0]] + next_split_line[1:]
            if len(split_line) > self.expected_length + 5:
                sys.stderr.write("Awkward formatting, probably from a comment:  Please fix line " + str(x) + "\n")
                sys.exit()
            self.read_answer_line(split_line)
            x += 1

    def get_data(self, line):
        line = line.strip()
        line = line.strip('"')
        split_line = line.split("\",\"")
        return split_line

    def read_answer_line(self, split_line):
        assignment_status_idx = self.idx_hash["AssignmentStatus"]
        assignment_status = split_line[assignment_status_idx]
        hit_id_idx = self.idx_hash["HITId"]
        hit_id = split_line[hit_id_idx]
        worker_id_idx = self.idx_hash["WorkerId"]
        worker_id = split_line[worker_id_idx]
        comment_idx = self.idx_hash["Answer.comment"]
        comment = split_line[comment_idx]
        if comment != "" and comment != "n/a" and comment != "&nbsp;":
            self.comments[hit_id][worker_id] = comment
        # If we've already processed, ignore.
        if assignment_status == "Rejected" or (self.screening and assignment_status == "Approved"):
            return
        for tweet in ("tweet0", "tweet1", "tweet2", "tweet3", "tweet4", "tweet5"):
            o_sent_idx = self.idx_hash["Answer." + tweet + "_sentiment"]
            tweet_id_idx = self.idx_hash["Input." + tweet + "_id"]
            before_idx = self.idx_hash["Input." + tweet + "_before_entity"]
            after_idx = self.idx_hash["Input." + tweet + "_after_entity"]
            entity_idx = self.idx_hash["Input." + tweet + "_entity"]
            tweet_id = split_line[tweet_id_idx]
            tweet_ne = split_line[entity_idx]
            tweet_sentiment = split_line[o_sent_idx]
            tweet_before = split_line[before_idx]
            tweet_after = split_line[after_idx]
            self.processed_tweet_ids[tweet_id] += 1
            if tweet_sentiment == "none":
                not_lang_idx = self.idx_hash["Answer." + tweet + "_notlang"]
                not_lang = split_line[not_lang_idx]
                if not_lang == "on":
                    if self.screening:
                        sys.stderr.write("\n" + hit_id + " " + worker_id + " has non-spanish tweet or bad entity; check:\n")
                        sys.stderr.write(tweet_id + ": " + tweet_before + " **" + tweet_ne + "** " + tweet_after + "\n")
                    self.answers[hit_id][worker_id][(tweet_id, tweet_ne, tweet_before, tweet_after)] = None
                    continue
                else:
                    # Shouldn't happen; HIT checks for this before allowing submit.
                    sys.stderr.write("Warning:  Worker " + worker_id + " skipped a tweet, setting sentiment to 'neutral'.\n")
            else:
                if not self.NEs_given:
                    NE_idx = self.idx_hash["Answer.NE" + tweet[-1]]
                    NE_label = split_line[NE_idx]
                    if NE_label == "persona":
                        NE_label = "PERSON"
                    elif NE_label.startswith("organizac"):
                        NE_label = "ORGANIZATION"
                    self.answers[hit_id][worker_id][(tweet_id, tweet_ne, tweet_before, tweet_after)] = (tweet_sentiment, NE_label)
                    continue
            self.answers[hit_id][worker_id][(tweet_id, tweet_ne, tweet_before, tweet_after)] = tweet_sentiment
    
    def get_processed_hits(self):
        return self.processed_tweet_ids

    def get_answers(self):
        return self.answers

    def get_comments(self):
        return self.comments
        
