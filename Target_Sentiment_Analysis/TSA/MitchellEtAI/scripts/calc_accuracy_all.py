import sys
import re
from copy import copy
from collections import defaultdict
from optparse import OptionParser

parser = OptionParser(usage="%prog test predictions [OPTIONS]")
parser.add_option("-l", "--linear", dest="linear", action="store_true", default=False,
                  help="Predictions are in linear format")
parser.add_option("-c", "--with-confidence",
                  action="store_true", dest="with_confidence", default=False,
                  help="Print out posteriors")
parser.add_option("-v", "--verbose", action="store_true", dest="verbose", default=False)
parser.add_option("-t", "--top-5", action="store", dest="top_5", default=False)
(options, args) = parser.parse_args()


global wanted_NEs

wanted_NEs = ("B", "I", "B_VOLITIONAL", "I_VOLITIONAL", "B_ORGANIZATION", "I_ORGANIZATION", "B_PERSON", "I_PERSON", "Bsentiment", "Bpositive", "Bnegative", "Bneutral", "I", "Inegative", "Ipositive", "Isentiment", "Ineutral")

if len(sys.argv) < 2:
    print "Usage:  python calc_accuracy.py test predictions"
    sys.exit()
test = open(sys.argv[1], "r")

top_5_list = []
if options.top_5:
    top_5_fid = open(options.top_5, "r").readlines()
    for line in top_5_fid:
        top_5_list += [line.strip()]

by_example = False
predicted_NE_sent = open(sys.argv[2], "r")

if options.with_confidence:
    conf_file = open("acc_conf.csv", "w+")

def norm_answers(answers):
    for example in answers:
        for RV in answers[example]:
            val = answers[example][RV]
            if isinstance(val, tuple):
                val = val[0]
            if val.startswith("I"):
                start_candidate = RV
                #if linear:
                #    pass
                #else:
                desired_val = re.sub("^I", "B", val)
                prev_num = int(RV.strip("W")) - 1
                if prev_num < 0:
                    prev_val = "START"
                else:
                    prev_RV = "W" + str(prev_num)
                    try:
                        prev_val = answers[example][prev_RV]
                        if isinstance(prev_val, tuple):
                            prev_val = prev_val[0]
                    except KeyError:
                        # Default case; when this is hidden.
                        prev_val = "O"
                # While we're not yet at the start of the NE span...
                while prev_val == val:
                    # Move backwards to find the start.
                    start_candidate = prev_RV
                    prev_num = int(prev_RV.strip("W")) - 1
                    if prev_num < 0:
                        prev_val = "START"
                    else:
                        prev_RV = "W" + str(prev_num)
                        try:
                            prev_val = answers[example][prev_RV]
                            if isinstance(prev_val, tuple):
                                prev_val = prev_val[0]
                        except KeyError:
                            prev_val = "O"
                # If the value before this one isn't the NE start,
                # then normalize this guy to make it the start.
                if options.linear:
                    if prev_val[0] != "B":
                        sys.stderr.write("Example " + str(example) + ": Changing " + start_candidate + " to Bneutral " + "\n")
                        answers[example][start_candidate] = "Bneutral"
                else:
                    if prev_val != desired_val: 
                        sys.stderr.write("Example " + str(example) + ": Changing " + start_candidate + " to " + desired_val + "\n")
                        answers[example][start_candidate] = desired_val
    return answers

def get_predicted(predicted, answers=defaultdict(lambda: defaultdict(defaultdict))):
    global wanted_NEs

    example = 0
    for line in predicted:
        line = line.strip()
        if line.startswith("//"):
            continue
        elif line.startswith("example"):
            example += 1
            continue
        else:
            split_line = line.split("=")
            RV = split_line[0]
            value = split_line[1]
            split_value = value.split()
            prob = None
            if len(split_value) > 1:
                value = split_value[0]
                prob = split_value[1]
            else:
                if options.with_confidence:
                    sys.stderr.write("Predictions file does not contain confidence scores.\n")
                    sys.stderr.write("Exiting...\n")
                    sys.exit()
            # If it's an output prediction, save it.
            if RV[-1] == "o":
                RV = RV[:-1]
                word_idx = int(RV[1:])
                if RV[0] == "W":
                    if options.linear:
                        answers[example][word_idx]["NE"] = value[0]
                        answers[example][word_idx]["sentiment"] = value[1:]
                    else:
                        if value in wanted_NEs:
                            answers[example][word_idx]["NE"] = value[0]
                        else:
                            answers[example][word_idx]["NE"] = "O"
                else:
                    answers[example][word_idx]["sentiment"] = value
    # Uncomment to norm the answers.
    #answers = norm_answers(answers)
    return answers

def get_observed(observed):
    global wanted_NEs
    
    example = 0
    go = False
    observations = {}
    desired_RVs = defaultdict(defaultdict)
    for line in observed:
        line = line.strip()
        if line.startswith("//"):
            go = True
            continue
        elif line.startswith("example"):
            example += 1
            observations[example] = defaultdict(defaultdict)
            continue
        elif line.startswith("features:"):
            go = False
        elif line == "":
            continue
        else:
            if go:
                split_line = line.split()
                # If this is an input variable, don't evaluate it.
                if len(split_line) == 3:
                    continue
                line = split_line[1].strip(";")
                split_line = line.split("=")
                # If this is a hidden variable, don't evaluate it.
                if len(split_line) == 1:
                    continue
                RV = split_line[0]
                value = split_line[1]
                word_idx = int(RV[1:])
                if RV[0] == "W":
                    if value[1:] == "neutral":
                        sys.stderr.write("Want joint file as test file.\n")
                        sys.exit()
                    #if linear:
                    #    observations[example][word_idx]["NE"] = value[0]
                    #    observations[example][word_idx]["sentiment"] = value[1:]
                    #else:
                    if value in wanted_NEs:
                        observations[example][word_idx]["NE"] = value[0]
                    else:
                        observations[example][word_idx]["NE"] = "O"
                else:
                    observations[example][word_idx]["sentiment"] = value
            elif options.top_5:
                for word in top_5_list:
                    if line.startswith("word_id_" + word + "("):
                        desired_RV = re.findall("\(([^\)]+)\)", line)
                        word_idx = int(desired_RV[0].strip('W'))
                        desired_RVs[example][word_idx] = {}
    if options.top_5:
        new_observations = defaultdict(lambda: defaultdict(defaultdict))
        for example in observations:
            for word_idx in observations[example]:
                if word_idx in desired_RVs[example]:
                    new_observations[example][word_idx] = copy(observations[example][word_idx])
        observations = new_observations
    return observations


def split_NE_sentiment(input_NE):
    NE_tag = input_NE[0]
    sent_tag = input_NE[1:]
    if NE_tag == "I":
        sent_tag = ""
    return (NE_tag, sent_tag)
    

def compare_observed_to_predicted(observed, predicted):
    #print observed
    #print predicted
    #sys.exit()
    #NE_matrix = {"B":{"B":0, "I":0, "O":0}, "I":{"B":0, "I":0, "O":0}, "O":{"B":0, "I":0, "O":0}}
    #sentiment_matrix = {"":{"":0}, "positive":{"positive":0, "negative":0, "neutral":0}, "negative":{"positive":0, "negative":0, "neutral":0}, "neutral":{"positive":0, "negative":0, "neutral":0}}
    acc_hash = {"true":0.0, "false":0.0}
    joint_hash = {"true":0.0, "false":0.0}
    sentiment_hash = {"true positive":0.0, "true negative":0.0, "false positive":0.0, "false negative":0.0}
    NE_hash = {"true positive":0.0, "true negative":0.0, "false positive":0.0, "false negative":0.0}
    wanted_sentiments = ("positive", "negative", "Bpositive", "Bnegative", "Bsentiment", "sentiment")
    acc_list = []
    conf_list = []
    example_hash = {}
    for example in sorted(observed):
        if options.verbose: print example
        example_hash[example] = {"true":0, "false":0}
        sent_true = 0.0
        for word in sorted(observed[example]):
            observed_NE = observed[example][word]["NE"]
            predicted_NE = predicted[example][word]["NE"]
            try:
                observed_sentiment = observed[example][word]["sentiment"]
            except KeyError:
                observed_sentiment = ""
            try:
                predicted_sentiment = predicted[example][word]["sentiment"]
            except KeyError:
                predicted_sentiment = ""

            #print observed_NE, predicted_NE
            #print observed_sentiment, predicted_sentiment
            if observed_sentiment == predicted_sentiment:
                sent_true += 1
            #NE_matrix[observed_NE][predicted_NE] += 1
            #sentiment_matrix[observed_sentiment][predicted_sentiment] += 1
            # All
            if observed_NE == predicted_NE and observed_sentiment == predicted_sentiment:
                acc_hash["true"] += 1
                #both_true += 1
                if options.verbose: print "Perfect match!", observed_NE, observed_sentiment
            else:
                acc_hash["false"] += 1
                #both_false += 1
                if options.verbose: print "Imperfect match", observed_NE + ":" + observed_sentiment, " --- ", predicted_NE + ":" + predicted_sentiment
            # NEs
            if observed_NE == predicted_NE and predicted_NE in wanted_NEs:
                NE_hash["true positive"] += 1
                if options.verbose: print "(tp) NE match:", observed_NE, predicted_NE
            if observed_NE == predicted_NE and predicted_NE not in wanted_NEs:
                NE_hash["true negative"] += 1
                if options.verbose: print "(tn) NE match:", observed_NE, predicted_NE
            if observed_NE != predicted_NE and predicted_NE in wanted_NEs:
                NE_hash["false positive"] += 1
                if options.verbose: print "(fp) NE mismatch:", observed_NE, predicted_NE
            if observed_NE != predicted_NE and predicted_NE not in wanted_NEs:
                NE_hash["false negative"] += 1
                if options.verbose: print "(fn) NE mismatch:", observed_NE, predicted_NE
            if observed_NE[0] == "B":
                # Sentiment
                if observed_sentiment == predicted_sentiment and predicted_sentiment in wanted_sentiments:
                    sentiment_hash["true positive"] += 1
                    if options.verbose: print "(tp) sentiment match:", observed_sentiment, predicted_sentiment
                if observed_sentiment == predicted_sentiment and predicted_sentiment not in wanted_sentiments:
                    sentiment_hash["true negative"] += 1
                    if options.verbose: print "(tn) sentiment match:", observed_sentiment, predicted_sentiment
                if observed_sentiment != predicted_sentiment and predicted_sentiment in wanted_sentiments:
                    sentiment_hash["false positive"] += 1
                    if options.verbose: print "(fp) sentiment mismatch:", observed_sentiment, predicted_sentiment
                if observed_sentiment != predicted_sentiment and predicted_sentiment not in wanted_sentiments:
                    sentiment_hash["false negative"] += 1
                    if options.verbose: print "(fn) sentiment mismatch:", observed_sentiment, predicted_sentiment
                # Joint
                if observed_sentiment == predicted_sentiment and observed_NE == predicted_NE:
                    joint_hash["true"] += 1
                    if options.verbose: print "Joint match:", observed_NE, observed_sentiment
                else:
                    joint_hash["false"] += 1
                    if options.verbose: print "Joint mismatch:", observed_NE + ":" + observed_sentiment, " --- ", predicted_NE + ":" + predicted_sentiment
            elif predicted_NE[0] == "B": #and predicted_sentiment != "": # Last part goes without saying, eh?
                if observed_sentiment == predicted_sentiment and observed_NE == predicted_NE:
                    joint_hash["true"] += 1
                    if options.verbose: print "Joint match:", observed_NE, observed_sentiment
                else:
                    joint_hash["false"] += 1
                    if options.verbose: print "Joint mismatch:", observed_NE + ":" + observed_sentiment, " --- ", predicted_NE + ":" + predicted_sentiment
        example_hash[example]["true"] = sent_true
        #example_hash[example]["false"] = both_false

    if options.with_confidence:
        for example in example_hash:
            print str(example_hash[example]["true"]) + ",",

    acc_true = acc_hash["true"]
    acc_false = acc_hash["false"]
    joint_true = joint_hash["true"]
    joint_false = joint_hash["false"]

    #Specificity: tn / (tn + fp)
    print "Accuracy:", round(acc_true/(acc_true + acc_false), 4)
    print "Joint:", round(joint_true/(joint_true + joint_false), 4)
    print "NE precision:", round(NE_hash["true positive"]/(NE_hash["true positive"] + NE_hash["false positive"]), 4)
    print "NE recall:", round(NE_hash["true positive"]/(NE_hash["true positive"] + NE_hash["false negative"]), 4)
    print "NE specificity:", round(NE_hash["true negative"]/(NE_hash["true negative"] + NE_hash["false positive"]), 4)

    try:
        print "Sentiment precision:", round(sentiment_hash["true positive"]/(sentiment_hash["true positive"] + sentiment_hash["false positive"]), 4)
    except ZeroDivisionError:
        print "Incalculable (tp + fp = 0.0)"

    try:
        print "Sentiment recall:", round(sentiment_hash["true positive"]/(sentiment_hash["true positive"] + sentiment_hash["false negative"]), 4)
    except ZeroDivisionError:
        print "Incalculable (tp + fn = 0.0)"

    print "Sentiment specificity:", round(sentiment_hash["true negative"]/(sentiment_hash["true negative"] + sentiment_hash["false positive"]), 4)
    if options.top_5:
        print "Subjectivity accuracy:", round((sentiment_hash["true negative"] + sentiment_hash["true positive"])/(sentiment_hash["true negative"] + sentiment_hash["true positive"] + sentiment_hash["false negative"] + sentiment_hash["false positive"]), 4)
        print sentiment_hash
    #print NE_matrix
    #total = 0.0
    #for obs in ("B", "I", "O"):
    #    total += sum(NE_matrix[obs].values())
    #for obs in ("B", "I", "O"):
    #    for pred in ("B", "I", "O"):
    #        print obs, pred, NE_matrix[obs][pred], round(NE_matrix[obs][pred]/total, 4)
    ##print sentiment_matrix
    #total = 0.0
    #for obs in ("positive", "negative", "neutral"):
    #    total += sum(sentiment_matrix[obs].values())
    #for obs in ("positive", "negative", "neutral"):
    #    for pred in ("positive", "negative", "neutral"):
    #        print obs, pred, sentiment_matrix[obs][pred], round(sentiment_matrix[obs][pred]/total, 4)

predicted = get_predicted(predicted_NE_sent)
observed = get_observed(test)
compare_observed_to_predicted(observed, predicted)
