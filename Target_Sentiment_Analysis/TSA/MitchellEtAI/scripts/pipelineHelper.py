import sys
import re
from collections import defaultdict

class Pipeline:
    def __init__(self, var_prefix="W", ex_delim="example", feat_delim="feature", comment="//"):
        self.var_prefix = var_prefix
        self.ex_delim = ex_delim
        self.feat_delim = feat_delim
        self.comment = comment
 
    def norm_answers(self, answers):
        """ Norms BIO encoding scheme, so any span that starts
            with an "I" is changed to start with a "B". """
        for example in answers:
            for RV in answers[example]:
                val = answers[example][RV]
                if val.startswith("I"):
                    desired_val = re.sub("^I", "B", val)
                    one_before_last = RV
                    prev_RV = self.var_prefix + str(int(RV.strip(self.var_prefix)) - 1)
                    if prev_RV in answers[example]:
                        prev_val = answers[example][prev_RV]
                        while prev_val == val:
                            one_before_last = prev_RV
                            one_before_last_val = prev_val
                            prev_RV = self.var_prefix + str(int(prev_RV.strip(self.var_prefix)) - 1)
                            if prev_RV in answers[example]:
                                prev_val = answers[example][prev_RV]
                            else:
                                prev_val = None
                        if prev_val != desired_val:
                            sys.stderr.write("Example " + str(example) + ": Changing " + one_before_last + " to " + desired_val + "\n")
                            answers[example][one_before_last] = desired_val
                    else:
                        answers[example][RV] = desired_val
        return answers


    def read_NE_predictions(self, NE_predictions):
        """ Reads in an ERMA-formatted predictions file. """
        NE_hash = {}
        # NOTE:  ERMA actually starts numbering at 0,
        # so to match its output, this should be -1.
        example = 0
        for line in NE_predictions:
            strip_line = line.strip()
            if strip_line == "":
                continue
            if line.startswith(self.comment):
                continue
            if line.startswith(self.ex_delim):
                example += 1
                NE_hash[example] = {}
                continue
            split_ass = strip_line.split("=")
            RV = split_ass[0][:-1]
            val = split_ass[1]
            val = val.split()
            if len(val) > 1:
                val = val[0]
            NE_hash[example][RV] = val
        NE_hash = self.norm_answers(NE_hash)
        return NE_hash
        
        
    def combine_NEs_with_sent(self, test_hash, NE_predictions):
        """ Combines NE predictions with sentiment model """
        combined_hash = {}
        if sorted(test_hash.keys()) != sorted(NE_predictions.keys()):
            sys.stderr.write("Error:  Sentences from pipeline stages mismatched\n.")
            sys.exit()
        for tweet_id in test_hash:
            combined_hash[tweet_id] = {}
            for (RV, label, sent) in test_hash[tweet_id]:
                pred_label = NE_predictions[tweet_id][RV]
                # Change the gold-standard label to the predicted label
                combined_hash[tweet_id][(RV, pred_label, sent)] = test_hash[tweet_id][(RV, label, sent)]
        return combined_hash