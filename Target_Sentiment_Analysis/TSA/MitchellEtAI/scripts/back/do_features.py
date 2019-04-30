#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import glob
import re
import codecs
import unicodedata
import getSignature as g
from optparse import OptionParser
import pipe_NE_predictions_to_sent_in as pipe_class
from collections import defaultdict

global known_words
global all_features
global all_types
global all_sent
global punc
global string_map
global stops
global prepositions
global determiners
global linear
global joint
global pipeline_NE
global pipeline_sent
global ipa
global vowels
global isUpperCase
global UniqueID
global curse_words
global syllable_structure
global language
global just_test
global extra_features
global horizon_feature_hash

train = open(sys.argv[1])
test = open(sys.argv[2])

UniqueID = "## Tweet"

usage = "%prog train_file test_file [OPTIONS]"
parser = OptionParser(usage=usage)
parser.add_option("-u", "--unique-id", type="string", help="Unique identifier for each sentence.", default="## Tweet", dest="UniqueID")
parser.add_option("-j", "--joint", dest="joint", help="Specify joint model.", default=False, action="store_true")
parser.add_option("-l", "--linear", dest="linear", help="Specify linear model.", default=False, action="store_true")
parser.add_option("-n", "--pipeline-NE", dest="pipeline_NE", help="Specify pipeline NE model.", default=False, action="store_true")
parser.add_option("-s", "--pipeline-sent", dest="pipeline_NE_predictions", help="Specify pipeline sent model; takes as input NE predictions from last step of the pipeline.", action="store", default=False)
parser.add_option("--noID", dest="noID", help="Mask lexical identity.", action="store_true", default=False)
parser.add_option("--sent-links", dest="SentSent", help="Specify whether to include links between sentiment variables.", action="store_true", default=False)
parser.add_option("--hidden-sent", dest="HiddenSent", help="Specify whether to include sentiment as latent variables in cases without a sentiment assignment.", action="store_true", default=False)
parser.add_option("--no-cursing", dest="no_cursing", help="Specifying whether to include curse words.", action="store_true", default=False)
parser.add_option("--no-polarity", dest="no_polarity", help="Collapse positive/negative into one category.", action="store_true", default=False)
parser.add_option("--no-sentiment", dest="no_sentiment", help="Just predict volitional entities", action="store_true", default=False)
parser.add_option("--volitional", dest="volitional", help="Collapse Person/Org into volitional", action="store_true", default=False)
parser.add_option("--just-test", dest="just_test", help="Just make a test file.", action="store_true", default=False)
parser.add_option("--note", dest="note", help="Note to append to files.", action="store", default=False)
parser.add_option("--language", dest="language", help="Set the language ('es' or 'en').", action="store", default="es")
(options, args) = parser.parse_args()
UniqueID = options.UniqueID

isUpperCase = re.compile("[A-ZÁÉÍÓÚÜÑ]", re.UNICODE)
NENE = True
#NoTheresa = False
pipeline_NE = False
pipeline_sent = False
withSent = True
#else:
#    withSent = False
bad_words_file = open(options.language + "/feature_files/bad_words", "r").readlines()
good_words_file = open(options.language + "/feature_files/good_words", "r").readlines()
curse_words_file = open(options.language + "/feature_files/curse_words", "r").readlines()
prepositions_file = open(options.language + "/feature_files/prepositions", "r").readlines()
determiners_file = open(options.language + "/feature_files/determiners", "r").readlines()
syllables_file = open(options.language + "/feature_files/syllables", "r").readlines()
other_feature_files = glob.glob(options.language + "/feature_files/*.txt")
extra_features = {}

volitional = options.volitional
noID = options.noID
SentSent = options.SentSent
HiddenSent = options.HiddenSent
no_cursing = options.no_cursing
no_sentiment = options.no_sentiment
linear = options.linear
no_polarity = options.no_polarity
language = options.language
just_test = options.just_test
if linear:
    withSent = False
joint = options.joint
if joint:
    #SentSent = True
    HiddenSent = True
    SentSent = True
pipeline_NE = options.pipeline_NE
if pipeline_NE:
    withSent = False
pipeline_NE_predictions = options.pipeline_NE_predictions
# If we have an NE predictions file we're sticking in,
# We're on the second stage of the pipeline
if pipeline_NE_predictions:
    pipeline_sent = True
    NENE = False
else:
    pipeline_sent = False
note = options.note
if not note:
    note = ""
stops = ("#", ".", ",", "?", ":", ";", "-", "!")
num = 3

all_features = {}
all_types = {}
all_sent = {}
horizon_feature_hash={}
known_words = defaultdict(int)
punc = re.compile("[^A-Za-z0-9_ÁÉÍÓÚÜÑáéíóúüñ]")
string_map = {".":"_P_" , ",":"_C_", "'":"_A_", "%":"_PCT_", "-":"_DASH_",
    "$":"_DOL_", "&":"_AMP_", ":":"_COL_", ";":"_SCOL_", "\\":"_BSL_"
    , "/":"_SL_", "`":"_QT_", "?":"_Q_", u"¿":"_QQ_", "=":"_EQ_", "*":"_ST_",
    "!":"_E_", u"¡":"_EE_", "#":"_HSH_", "@":"_AT_", "(":"_LBR_", ")":"_RBR_"
    , "\"":"_QT0_", u"Á":"_A_ACNT_", u"É":"_E_ACNT_", u"Í":"_I_ACNT_",
    u"Ó":"_O_ACNT_", u"Ú":"_U_ACNT_", u"Ü":"_U_ACNT0_", u"Ñ":"_N_ACNT_",
    u"á":"_a_ACNT_", u"é":"_e_ACNT_", u"í":"_i_ACNT_", u"ó":"_o_ACNT_",
    u"ú":"_u_ACNT_", u"ü":"_u_ACNT0_", u"ñ":"_n_ACNT_", u"º":"_deg_ACNT_"}


def read_known_words(known_words_file):
    word_list = []
    for line in known_words_file:
        line = line.strip()
        if line == "":
            continue
        line = line.lower()
        word_list += [line.decode("utf8")]
    return word_list


def read_syllables_file(syllables_file):
    onsets = [""]
    vowels = []
    codas = [""]
    for line in syllables_file:
        if line.startswith("## onsets"):
            do_onset = True
            do_vowel = False
            do_coda = False
        elif line.startswith("## vowels"):
            do_onset = False
            do_vowel = True
            do_coda = False
        elif line.startswith("## codas"):
            do_onset = False
            do_vowel = False
            do_coda = True
        else:
            if do_onset:
                onsets += [line.strip()]
            elif do_vowel:
                vowels += [line.strip()]
            elif do_coda:
                codas += [line.strip()]
    return (onsets, vowels, codas)

def escape(s):
    global string_map
    #print "before: ", s, type(s)
    if type(s) is str:
        s = s.decode('utf-8')
    for val, repl in string_map.iteritems():
        s = s.replace(val,repl)
    #print "after: ", s, type(s)
    return s

def norm_digits(s):
    return re.sub("\d+","0",s, re.UNICODE)

def clean(s):
    global punc
    s = norm_digits(s.lower())
    s = escape(s)
    return punc.sub("_",s, re.UNICODE)


def make_links(tweet_words):
    global isUpperCase

    s_chains = defaultdict(defaultdict)
    for (word, n) in tweet_words:
        # If it's capitalized...
        if isUpperCase.search(word[0]):
            possible_NE = word.lower()
            for (other_word, other_n) in tweet_words:
                if other_n == n:
                    continue
                if other_word.lower() == possible_NE and (other_n, n) not in s_chains[possible_NE]:
                    s_chains[possible_NE][(n, other_n)] = {}
    return s_chains

def read_file(in_file):
    global known_words
    global UniqueID

    tweet_words = {}
    skip_chains = {}
    word_hash = {}
    tweet_id = 0
    for line in in_file:
        line = line.strip()
        #print line
        if line == "":
            continue
        elif line.startswith(UniqueID):
            # Grab the new set of features and RVs.
            split_line = line.split()
            skip_chains[tweet_id] = make_links(tweet_words)
            #print skip_chains
            #sys.exit()
            #tweet_id = split_line[-1]
            tweet_id += 1
            word_hash[tweet_id] = {}
            tweet_words= {}
            n = 0
        elif line.startswith("SentimentLex"):
            continue
        else:
            split_line = line.split("\t")
            if len(split_line) != 8:
                sys.stderr.write(str(tweet_id) + "\n")
                sys.stderr.write(line + "\n")
                sys.stderr.write("Skipping weird line.\n")
                sys.exit()
                #print line
                continue
            word = split_line[0].strip()
            # This is the NE.
            NE = split_line[1].strip()
            if volitional and not linear: # Linear should never be specified in this case anyway.
                if NE.startswith("B") or NE.startswith("I"):
                    if not NE.endswith("ORGANIZATION") and not NE.endswith("PERSON"):
                        NE = "O"
                    else:
                        NE = re.sub("-\S+", "-VOLITIONAL", NE)
            NE = re.sub("-", "_", NE)
            #col2 = "_col2_" + clean(split_line[2].strip())
            brown_cluster5 = "_brown5_" + split_line[2].strip()
            # Jerboa features
            jerboa = "_jerboa_" + clean(split_line[3].strip())
            # NEs from David's thing?
            #col4 = "_col4_" + clean(split_line[4].strip())
            brown_cluster3 = "_brown3_" + split_line[4].strip()
            if len(split_line) > 5:
                # This is the sentiment column.
                prior_polarity = "_sent_" + clean(split_line[5].strip())
                #if not NoTheresa: 
                prior_ther_polarity = "_sent_ther_" + re.sub(":", "_", split_line[6].strip()).lower()
                #else:
                #    prior_ther_polarity = "_"
                # This is the sentiment towards the entity.
                targeted_polarity = split_line[-1].strip()
                if no_polarity and (targeted_polarity == "positive" or targeted_polarity == "negative"):
                    targeted_polarity = "sentiment"
                if no_sentiment and targeted_polarity != "_":
                    targeted_polarity = "volitional"
            else:
                prior_polarity = "_"
                theresa_polarity = "_"
                targeted_polarity = "_"
            word_hash[tweet_id][n] = (word, NE, brown_cluster5, jerboa, brown_cluster3, prior_polarity, prior_ther_polarity, targeted_polarity)
            try:
                esc_word = unicode(word, 'unicode-escape')
            except UnicodeDecodeError:
                try:
                    esc_word = unicode(word)
                except UnicodeDecodeError:
                    esc_word = unicodedata.normalize('NFKD', unicode(word, 'utf8')).encode('ASCII','ignore')
            known_words[esc_word] += 1
            tweet_words[(esc_word, n)] = {}
            n += 1
    skip_chains[tweet_id] = make_links(tweet_words)
    return (word_hash, skip_chains)


def bin_wl(length):
    if length < 5:
        group = 1
    elif length < 8:
        group = 2
    else:
        group = 3
    feature = "_word_length_bin_" + str(group)
    return feature

def bin_pos(pos, sent_len):
    # Position in sentence.
    div = sent_len/3.0
    group = None
    if pos <= div:
        group = 1
    elif pos <= div * 2:
        group = 2
    elif pos <= sent_len:
        group = 3
    if not group:
        sys.stderr.write("Sentence position incorrectly assigned.\n")
    feature = "_message_position_bin_" + str(group)
    return feature
    
def bin_msg(sent_len):
    if sent_len <= 5:
        group = 1
    elif sent_len <= 10:
        group = 2
    else:
        group = 3
    feature = "_message_length_bin_" + str(group)
    return feature

def get_last_letters(word):
    feature_set = {}
    if len(word) < 8:
        word_num = len(word)
    else:
        word_num = 8
    for n in range(word_num):
        last_letters = word[-1 * n:]
        feature_set[last_letters] = {}
    return feature_set

def syllabize(word):
    #global vowels
    global syllable_structure
    vowels = ''.join(syllable_structure[1])
    word = word.lower()
    split_word = word.split("-")
    sylls = re.findall("([^" + vowels + "]*?[" + vowels + "])", word, re.UNICODE) 
    sylls_tmp = re.findall("[^" + vowels + "]*?$", word, re.UNICODE)
    if sylls !=[] and sylls_tmp != []:
        sylls[-1] += sylls_tmp[0]
    return sylls

def get_sonority(word):
#    global ipa
#    global vowels
#    allowed_sequences = []
#    allowed_beginnings = ['pav', 'pv', 'fpv', 'fv', 'av', 'nv', 'v', 'fav']
#    allowed_endings = ['n', 'a', 's', 'p', 'f']
#    for allowed_beginning in allowed_beginnings:
#        allowed_sequences += [allowed_beginning]
#        for allowed_ending in allowed_endings:
#            allowed_sequences += [allowed_beginning + allowed_ending]
    global syllable_structure
    onsets = syllable_structure[0]
    vowels = ''.join(syllable_structure[1])
    codas = syllable_structure[2]
    syllables = []
    if not re.search("[A-Za-z]", word):
        return None
    split_word = word.split("-")
    for word in split_word:
        if word == "":
            continue
        syllables_tmp = syllabize(word)
        if syllables_tmp == []:
            #print "can't syllabize "
            #print word
            return "_cannot_syllabize"
        syllables += syllables_tmp
    last_pattern = None
    last_coda = ""
    for syll in syllables:
        #print syll
        split_syllable = re.findall("([^" + vowels + "]*)([" + vowels + "])([^" + vowels + "]*)", syll)
        split_syllable = split_syllable[0]
        #print "Split syllable is ", split_syllable
        onset = split_syllable[0]
        coda = split_syllable[-1]
        if (onset == "" or onset in onsets) and (coda == "" or coda in codas):
            last_coda = coda
            continue
        else:
            while onset not in onsets or coda not in codas or last_coda not in codas:
                #print word, syll, last_coda, onset, coda
                try:
                    last_coda += onset[0]
                    onset = onset[1:]
                except IndexError:
                    #print "can't syllabize "
                    #print word
                    return "_cannot_syllabize"
            last_coda = coda
                #if x > 4:
                #    return "_cannot_syllabize"
    return None

def get_features(node, pos, sent_len, prev_prev_node, prev_node, next_node):
    global all_types
    global all_sent
    global stops
    global prepositions
    global determiners
    global linear
    global curse_words
    global horizon_feature_hash
    """ Node: (word, NE, brown_cluster5, jerboa, brown_cluster3, prior_polarity, theresa_polarity, targeted_polarity) """
    feature_tmp = {}
    has_no = False
    (word, NE, brown_cluster5, jerboa, brown_cluster3, prior_polarity, theresa_polarity, targeted_polarity) = node
    (prev_prev_word, prev_word, next_word) = ("", "", "")
    if prev_prev_node:
        prev_prev_word = prev_prev_node[0]
        if prev_prev_word.lower() == "no":
            has_no = True
    if prev_node:
        prev_word = prev_node[0]
        if prev_word.lower() == "no":
            has_no = True
        if prev_word[0] in stops:
            feature_tmp["_prev_in_stops"] = {}
    if next_node:
        next_word = next_node[0]
        if next_word[0] in stops:
            feature_tmp["_next_in_stops"] = {}
    # General features related to sentence
    # position and length.
    if pos == 0:
        feature_tmp["_is_first"] = {}
    elif pos == 1:
        feature_tmp["_is_second"] = {}
    elif pos == 2:
        feature_tmp["_is_third"] = {}
    if sent_len - pos - 1 == 0:
        feature_tmp["_is_last"] = {}
    elif sent_len - pos - 1 == 1:
        feature_tmp["_is_second_last"] = {}
    elif sent_len - pos - 2 == 2:
        feature_tmp["_is_third_last"] = {}
    position_feature = bin_pos(pos, sent_len)
    feature_tmp[position_feature] = {}
    message_feature = bin_msg(sent_len)
    feature_tmp[message_feature] = {}
    word_length_feature = bin_wl(len(word))
    feature_tmp[word_length_feature] = {}
    #sonority_feature = get_sonority(word)
    #feature_tmp[sonority_feature] = {}
    # This is the label we're predicting.
    if not linear:
        all_types[NE] = {}
    # Brown clusters
    feature_tmp[brown_cluster5] = {}
    feature_tmp[brown_cluster3] = {}
    # Jerboa features.
    feature_tmp[jerboa] = {}
    repeat = 0
    prev_char = None
    for char in word:
        if char == prev_char:
            repeat += 1
        if repeat >= 2:
            feature_tmp["_sent_has_repeat"] = {}
            break
    if "emoticon" in jerboa or "smiling" in jerboa or "frowning" in jerboa:
        if ")" in word or "D" in word or "]" in word:
            feature_tmp["_sent_jerb_happy"] = {}
        elif "(" in word or "[" in word:
            feature_tmp["_sent_jerb_sad"] = {}
    if "!" in word:
        feature_tmp["_sent_has_exclaim"] = {}
    if "!!" in word:
        feature_tmp["_sent_has_many_exclaim"] = {}
    if "..." in word:
        feature_tmp["_sent_has_ellipse"] = {}
    if "?" in word:
        feature_tmp["_sent_has_question"] = {}
    if "??" in word:
        feature_tmp["_sent_has_many_question"] = {}
    lower_word = word.lower().decode("utf8")
    if "haha" in lower_word or "jaja" in lower_word or "jeje" in lower_word or "hehe" in lower_word or "hihi" in lower_word or "jiji" in lower_word:
        feature_tmp["_sent_has_laugh"] = {}
    if re.search("^noo+$", lower_word):
        feature_tmp["_sent_has_noo"] = {}
    if lower_word == "no":
        feature_tmp["_sent_has_no"] = {}
    #if lower_word == "saludos":
    #    feature_tmp["_sent_has_saludos"] = {}
    if lower_word in bad_words:#== "mal" or lower_word == "malo" or lower_word == "mala":
        feature_tmp["_sent_has_mal"] = {}
    if lower_word in good_words: #== "bien" or lower_word == "bueno" or lower_word == "buena" or lower_word == "buen": #or lower_word == "grande" or lower_word == "gran":
        feature_tmp["_sent_has_bien"] = {}
    if lower_word == "my" or lower_word == "mis" or lower_word == "mi":
        feature_tmp["_sent_has_mi"] = {}
    if lower_word == "nunca" or lower_word == "nadie":
        feature_tmp["_sent_has_no_word"] = {}
    #if lower_word == "pero":
    #    feature_tmp["_sent_has_but"] = {}
    #if lower_word == "demasiado":
    #    feature_tmp["_sent_has_demasiado"] = {}
    if lower_word in curse_words and not no_cursing:
        feature_tmp["_sent_has_curse_word"] = {}
    #if options.language == "es" and (lower_word.endswith("ito") or lower_word.endswith("ita")):
    #    feature_tmp["_sent_has_diminutive"] = {}
    for (feature_file, feature_type) in extra_features:
        if feature_type == "Suff":
            for suff in extra_features[(feature_file, feature_type)]:
                #print lower_word, suff
                if lower_word.endswith(suff):
                    #print "adding feature for...suff", suff
                    feature_tmp["_sent_" + feature_file] = {}
        elif feature_type == "Prefix":
            for pref in extra_features[(feature_file, feature_type)]:
                if lower_word.startswith(pref):
                    #print "adding feature for...pref", pref
                    feature_tmp["_sent_" + feature_file] = {}
        if lower_word in extra_features[(feature_file, feature_type)]:
            #print "adding feature for...", lower_word
            feature_tmp["_sent_" + feature_file] = {}
    # Suff
    #if lower_word.startswith("des"):
    #    feature_tmp["_sent_has_des"] = {}
    #if lower_word == "amo" or lower_word == "querido" or lower_word == "querida" or lower_word == "quiero" or lower_word == "gusta" or lower_word == "encanta" or lower_word == "espero" or lower_word == "gustaria" or lower_word == "gustaría":
    #    if not has_no:
    #        feature_tmp["_sent_has_like"] = {}
        #else:
        #    feature_tmp["_sent_has_no_like"] = {}
    # Previous known sentiment
    # of this word
    feature_tmp[prior_polarity] = {}
    #if not NoTheresa:
    feature_tmp[theresa_polarity] = {}
    # This is the sentiment we're predicting.
    all_sent[targeted_polarity] = {}
    if prior_polarity[-1] != "_":
        #print prior_polarity
        feature_tmp["_is_sent"] = {}
    if theresa_polarity[-1] != "_": #and not NoTheresa:
        feature_tmp["_is_sent_ther"] = {}
    for feature in feature_tmp:
        if "sent_" in feature:
            horizon_feature_hash[feature] = {}
    #else:
    #    feature_tmp["_is_not_sent_ther_"] = {}
    # Specific lexical features for the word.
    try:
        esc_word = unicode(word, 'unicode-escape')
    except UnicodeDecodeError:
        try:
            esc_word = unicode(word)
        except UnicodeDecodeError:
            esc_word = unicodedata.normalize('NFKD', unicode(word, 'utf8')).encode('ASCII','ignore')
    lower_word = esc_word.lower()
    normalized_word = clean(esc_word)
    if (len(esc_word) == 3 or len(esc_word) == 4) and esc_word not in determiners:
        feature_tmp["_is_tli"] = {}
    if jerboa == "_jerboa_none":
        syll_feature = get_sonority(lower_word)
        if syll_feature:
            feature_tmp[syll_feature] = {}
    #print feat, type(feat)
    #last_letters_hash = get_last_letters(esc_word)
    #for last_letters in last_letters_hash:
    #    clean_last_letters = "_ending_" + clean(last_letters)
    #    clean_last_letters = unicodedata.normalize('NFKD', clean_last_letters).encode('ASCII','ignore')
    #    feature_tmp[clean_last_letters] = {}
    if esc_word not in known_words or known_words[esc_word] < 3:
        if not noID:
            oov = "_" + clean(g.getSignature(esc_word, language=language).unk)
            oov = unicodedata.normalize('NFKD', oov).encode('ASCII','ignore')
            feature_tmp[oov] = {}
    else:
        if not noID:
            word_id = "_id_" + clean(esc_word)
            word_id = unicodedata.normalize('NFKD', word_id).encode('ASCII','ignore')
            feature_tmp[word_id] = {}
    simple_features = g.getSignature(esc_word, language=language).simple_unk_features
    for s in simple_features:
        feature_tmp[s] = {}
    if lower_word in determiners:
        feature_tmp["_is_determiner"] = {}
    if lower_word in prepositions:
        feature_tmp["_is_preposition"] = {} #print feature_tmp
    #sys.exit()
    if "_INITC" in feature_tmp:
        if "_is_first" not in feature_tmp:
            feature_tmp["_INITC_NOT_FIRST"] = {}
        if "_is_determiner" in feature_tmp:
            feature_tmp["_INITC_DETERMINER"] = {}
    if "_CAPS" in feature_tmp:
        if "_cannot_syllabize" in feature_tmp:
            feature_tmp["_CAPS_CANNOT_SYLLABIZE"] = {}
        if "_is_tli" in feature_tmp:
            feature_tmp["_CAPS_IS_TLI"] = {}
        if "_cannot_syllabize" in feature_tmp and "_is_tli" in feature_tmp:
            feature_tmp["_CAPS_IS_TLI_CANNOT_SYLLABIZE"] = {}
    return feature_tmp


def get_features_with_prefix(features_tweet_RV, tmp_features, prefix, test):
    global all_features
    for feature in tmp_features:
        feature = prefix + feature
        if not test:
            all_features[feature] = all_features.setdefault(feature,0) + 1
            features_tweet_RV[feature] = {}
        elif feature in all_features:
            features_tweet_RV[feature] = {}
    return features_tweet_RV


def bin_sent(num_sent):
    features_tmp = {}
    if num_sent == 1:
        features_tmp["_has_one_sent"] = {}
    elif num_sent == 2:
        features_tmp["_has_two_sent"] = {}
    elif num_sent == 3:
        features_tmp["_has_three_sent"] = {}
    elif num_sent > 3:
        features_tmp["_has_alotta_sent"] = {}
    return features_tmp

def get_horizon_polarity(features_tweet_RV, features_tmp, n, test):
    global all_features
    global horizon_feature_hash
    horizon_features = horizon_feature_hash.keys()

    i = 0
    while i < n - 3:
        for horizon_feature in horizon_features:
            if horizon_feature in features_tmp[i]:
                feature = "_prev" + horizon_feature
                if not test:
                    features_tweet_RV[feature] = {}
                    all_features[feature] = all_features.setdefault(feature,0) + 1
                elif feature in all_features:
                    features_tweet_RV[feature] = {}
        i += 1
    if n < 3:
        i = 0
    else:
        i = n - 3
    while i < n:                                
        for horizon_feature in horizon_features:                    
            if horizon_feature in features_tmp[i]:
                feature = "_immediate_prev" + horizon_feature
                if not test:
                    features_tweet_RV[feature] = {}
                    all_features[feature] = all_features.setdefault(feature,0) + 1
                elif feature in all_features:
                    features_tweet_RV[feature] = {}
        i += 1
    i = n + 3
    while i < len(features_tmp):
        for horizon_feature in horizon_features:
            if horizon_feature in features_tmp[i]:
                feature = "_next" + horizon_feature
                if not test:
                    features_tweet_RV[feature] = {}
                    all_features[feature] = all_features.setdefault(feature,0) + 1
                elif feature in all_features:
                    features_tweet_RV[feature] = {}
        i += 1
    i = n
    while i < n + 3 and i < len(features_tmp):
        for horizon_feature in horizon_features:
            if horizon_feature in features_tmp[i]:
                feature = "_immediate_next" + horizon_feature
                if not test:
                    features_tweet_RV[feature] = {}
                    all_features[feature] = all_features.setdefault(feature,0) + 1
                elif feature in all_features:
                    features_tweet_RV[feature] = {}
        i += 1
    num_sent = 0
    for n in features_tmp:
        features = features_tmp[n]
        if "_is_sent" in features:
            num_sent += 1
    overall_sent_features =  bin_sent(num_sent)
    for overall_sent_feature in overall_sent_features:
        feature = "_sent_overall" + overall_sent_feature
        if not test:
            features_tweet_RV[feature] = {}
            all_features[feature] = {}
        elif feature in all_features:
            features_tweet_RV[feature] = {}
    return features_tweet_RV

def make_nodes(word_hash, test=False):
    global num
    features = {}
    has_rel_NE = {}
    for tweet_id in word_hash:
        #print "Looking at "
        #print word_hash[tweet_id]
        features[tweet_id] = {}
        features_tmp = {}
        n = 0
        while n < len(word_hash[tweet_id]):
            # (word, NE, jerboa, prior_polarity, prior_ther_polarity, targeted_polarity) = node
            node = word_hash[tweet_id][n]
            NE = node[1]
            # Don't include guys we don't need (I believe I'm screening this out earlier at this point.)
            if not test:
                if ("ORGANIZATION" in NE or "PERSON" in NE or "VOLITIONAL" in NE):
                    has_rel_NE[tweet_id] = {}
            else:
                has_rel_NE[tweet_id] = {}
            try:
                prev_node = word_hash[tweet_id][n-1]
            except KeyError:
                prev_node = False
            try:
                prev_prev_node = word_hash[tweet_id][n-2]
            except KeyError:
                prev_prev_node = False
            try:
                next_node = word_hash[tweet_id][n+1]
            except KeyError:
                next_node = False
            features_tmp[n] = get_features(node, n, len(word_hash[tweet_id]), prev_prev_node, prev_node, next_node)
            n += 1
        # Make sure this tweet has something we want to train on.
        # If not, continue
        if not test and (tweet_id not in has_rel_NE):
            features[tweet_id] = None
            continue
        n = 0
        while n < len(word_hash[tweet_id]):
            node = word_hash[tweet_id][n]
            RV = "W" + str(n)
            label = node[1]
            sent = node[-1]
            # Whether to have hidden sentiment beliefs
            # Do not encode features for hidden sentiment
            # If we're in the second stage of the pipeline sentiment
            if sent == "_" and (pipeline_sent and not HiddenSent):
                n += 1
                continue
            prefix = "word"
            features[tweet_id][(RV, label, sent)] = get_features_with_prefix({}, features_tmp[n], prefix, test)
            p = 0
            while p < num and n+p < len(word_hash[tweet_id]) - 1:
                p += 1
                ahead_prefix = "wordp" + str(p)
                features[tweet_id][(RV, label, sent)] = get_features_with_prefix(features[tweet_id][(RV, label, sent)], features_tmp[n+p], ahead_prefix, test)
            p = 0
            while p < num and n-p > 0:
                p += 1
                behind_prefix = "wordm" + str(p)
                features[tweet_id][(RV, label, sent)] = get_features_with_prefix(features[tweet_id][(RV, label, sent)], features_tmp[n-p], behind_prefix, test)
            features[tweet_id][(RV, label, sent)] = get_horizon_polarity(features[tweet_id][(RV, label, sent)], features_tmp, n, test)
            n += 1
        #print "features are " 
        #print features[tweet_id]
    return features


def make_all_features():
    global all_features
    keys = all_features.keys()
    for feature in keys:
        if all_features[feature] < 2:
            del all_features[feature]

def print_linear_variables(tt, RV, label, sent):
    if sent == "_":#or (not re.search("ORGANIZATION", label) and not re.search("PERSON", label)):
        # Hide it.
        if label == "B_ORGANIZATION" or label == "B_PERSON" or label == "B_VOLITIONAL":
            print >> tt, "NESENT " + RV + ";"
        elif label == "I_ORGANIZATION" or label == "I_PERSON" or label == "I_VOLITIONAL":
            print >> tt, "NESENT " + RV + "=I;"
        else:
            print >> tt, "NESENT " + RV + "=O;"
    else:
        # Sentiment only expressed on volitional entities.
        if label == "B_ORGANIZATION" or label == "B_PERSON" or label == "B_VOLITIONAL":
            NESENT = "B" + sent
        elif label == "I_ORGANIZATION" or label == "I_PERSON" or label == "I_VOLITIONAL":
            NESENT = "I" + sent
        print >> tt, "NESENT " + RV + "=" + NESENT + ";"

def print_joint_variables(tt, RV, label, sent, sent_chains):
    if label == "_":
        print >> tt, "NE " + RV + ";"
    else:
        print >> tt, "NE " + RV + "=" + label + ";"
    SV = "S" + RV[1:]
    if sent == "_":
        print >> tt, "SENT " + SV + ";"
    else:
        print >> tt, "SENT " + SV + "=" + sent + ";"
    sent_chains[int(SV[1:])] = {}
    return sent_chains

def print_pipe_NE_variables(tt, RV, label):
    if label == "_":
        print >> tt, "NE " + RV + ";"
    else:
        print >> tt, "NE " + RV + "=" + label + ";"

def print_pipe_sent_variables(tt, RV, label, sent, sent_chains):
    print >> tt, "NE " + RV + "=" + label + " in;"
    SV = "S" + RV[1:]
    if sent == "_" and HiddenSent:
        print >> tt, "SENT " + SV + ";"
        sent_chains[int(SV[1:])] = {}
    else:
        print >> tt, "SENT " + SV + "=" + sent + ";"
        sent_chains[int(SV[1:])] = {}
    return sent_chains

def print_skip_chain(tt, skip_chains):
        for chain in skip_chains:
            for skip_chain in skip_chains[chain]:
                n = skip_chain[0]
                n2 = skip_chain[1]
                RV1 = "W" + str(n)
                RV2 = "W" + str(n2)
                print >> tt, "skip_chain(" + RV1 + "," + RV2 + ");"


def print_sent_chain(tt, pipe_sent_vars):
            sent_vars = pipe_sent_vars.keys()
            sent_vars.sort()
            s = 0
            while s + 1 < len(sent_vars):
                s_num = sent_vars[s]
                s_num2 = sent_vars[s+1]
                print >> tt, "sent_chain(S" + str(s_num) + ",S" + str(s_num2) + ");"
                s += 1


def print_linear_features(tt, RV, label, sent, in_hash):
    for feature in sorted(in_hash[(RV, label, sent)]):
        if feature in all_features:
            print >> tt, feature + "(" + RV + ");"
    last_RV_num = int(RV.strip("W")) - 1
    if last_RV_num < 0:
        last_RV = "BEGIN"
    else:
        last_RV = "W" + str(last_RV_num)
    print >> tt, "link(" + last_RV + "," + RV + ");"

def print_joint_features(tt, RV, label, sent, in_hash):
    SV = "S" + RV[1:]
    for feature in sorted(in_hash[(RV, label, sent)]):
        if feature in all_features:
            print >> tt, feature + "(" + RV + ");"
            if "_sent_" in feature or "_id_" in feature:
                print >> tt, feature + "_link(" + RV + "," + SV + ");"
                print >> tt, feature + "_sent(" + SV + ");"
    print >> tt, "sent_link(" + RV + "," + SV + ");"
    last_RV_num = int(RV.strip("W")) - 1
    if last_RV_num < 0:
        last_RV = "BEGIN"
    else:
        last_RV = "W" + str(last_RV_num)
    print >> tt, "link(" + last_RV + "," + RV + ");"

def print_pipe_NE_features(tt, RV, label, sent, in_hash):
    for feature in sorted(in_hash[(RV, label, sent)]):
        if feature in all_features:
            print >> tt, feature + "(" + RV + ");"
    last_RV_num = int(RV.strip("W")) - 1
    if last_RV_num < 0:
        last_RV = "BEGIN"
    else:
        last_RV = "W" + str(last_RV_num)
    print >> tt, "link(" + last_RV + "," + RV + ");"

def print_pipe_sent_features(tt, RV, label, sent, in_hash, sent_chains):
    Snum = int(RV[1:])
    if Snum not in sent_chains:
        return
    SV = "S" + str(Snum)
    for feature in sorted(in_hash[(RV, label, sent)]):
        if feature in all_features:
            if "_sent_" in feature or "_id_" in feature:
                print >> tt, feature + "_link(" + RV + "," + SV + ");"
                print >> tt, feature + "_sent(" + SV + ");"
    print >> tt, "sent_link(" + RV + "," + SV + ");"

def print_train_test(in_hash, tt, skip_chains, test=False):
    global all_features
    global linear
    global withSent
    global pipeline_NE
    global pipeline_sent
    for tweet_id in sorted(in_hash):
        if not in_hash[tweet_id]:
            continue
        sent_chains = {}
        print >> tt, "//Tweet " + str(tweet_id)
        print >> tt, "example:"
        for (RV, label, sent) in sorted(in_hash[tweet_id]):
            if linear:
                print_linear_variables(tt, RV, label, sent)
            elif joint:
                sent_chains = print_joint_variables(tt, RV, label, sent, sent_chains)
            elif pipeline_NE:
                print_pipe_NE_variables(tt, RV, label)
            elif pipeline_sent:
                sent_chains = print_pipe_sent_variables(tt, RV, label, sent, sent_chains)
        print >> tt, "features:"
        if SentSent:
            print_sent_chain(tt, sent_chains)
        if not pipeline_sent:
            print_skip_chain(tt, skip_chains[tweet_id])
        for (RV, label, sent) in sorted(in_hash[tweet_id]):
            if linear:
                print_linear_features(tt, RV, label, sent, in_hash[tweet_id])
            elif joint:
                print_joint_features(tt, RV, label, sent, in_hash[tweet_id])
            elif pipeline_NE:
                print_pipe_NE_features(tt, RV, label, sent, in_hash[tweet_id])
            elif pipeline_sent:
                print_pipe_sent_features(tt, RV, label, sent, in_hash[tweet_id], sent_chains)

def print_linear_template(tt):
    global all_features
    global all_types
    global all_sent
    print >> tt, "types:"
    if no_sentiment:
        print >> tt, "NESENT:=[Bvolitional,I,O]"
    elif no_polarity:
        print >> tt, "NESENT:=[Bneutral,Bsentiment,I,Isentiment,Ineutral,O]"
    else:
        print >> tt, "NESENT:=[Bneutral,Bpositive,Bnegative,I,Ineutral,Ipositive,Inegative,O]" 
    print >> tt, ""
    print >> tt, "features:"
    if NENE:
        print >> tt, "link(NESENT,NESENT):=[*,*]"
        print >> tt, "skip_chain(NESENT,NESENT):=[*,*]"
    for feat in all_features:
        print >> tt, feat + "(NESENT):=[*]"

                
def print_template(tt):
    global all_features
    global all_types
    global all_sent
    global withSent
    global NENE
    global SentSent
    print >> tt, "types:"
    print >> tt, "NE:=[" + ",".join(all_types) + "]" 
    if withSent:
        if no_polarity:
            print >> tt, "SENT:=[sentiment,neutral]"
        else:
            print >> tt, "SENT:=[positive,negative,neutral]" #,both]"# + ",".join(all_sent) + "]"
    print >> tt, ""
    print >> tt, "features:"
    if not pipeline_sent:
        print >> tt, "skip_chain(NE,NE):=[*,*]"
    if SentSent:
        print >> tt, "sent_chain(SENT,SENT):=[*,*]"
    if NENE:
        print >> tt, "link(NE,NE):=[*,*]"
    #if withSent:
        #print >> tt, "sent_link(SENT,SENT):=[*,*]"
    if withSent:
        # True if joint or pipeline_sent
        print >> tt, "sent_link(NE,SENT):=[*,*]"
        # True in joint model
        if SentSent:
            print >> tt, "sent_sent_link(SENT,SENT):=[*,*]"
    for feat in all_features:
        #try:
        if not pipeline_sent:
            print >> tt, feat + "(NE):=[*]"
        if withSent:
            # True if joint or pipeline_sent:
            if "_sent_" in feat or "_id_" in feat:
                print >> tt, feat + "_link(NE,SENT):=[*,*]"
                print >> tt, feat + "_sent(SENT):=[*]"
            #print >> tt, feat + "_sent(SENT):=[*]"
        #except UnicodeEncodeError:
        #    sys.stderr.write("Encoding error.\n")
        #    sys.stderr.write(feat.encode('ascii', 'xmlcharrefreplace') + " printed as " + feat.encode('ascii', 'ignore'))
        #    sys.stderr.write("\n")
        #    print >> tt, feat.encode('ascii', 'ignore') + "(NE):=[*]"

def print_qsub(qsub, setting):
    global pipeline_sent

    #full_path = os.getcwd()
    qsub_str = "#!/bin/bash\n\n"
    qsub_str += "#$ -S /bin/bash\n"
    qsub_str += "#$ -cwd\n"
    qsub_str += "#$ -l num_proc=1,h_rt=36:00:00,h_vmem=40g,mem_free=40g\n"
    qsub_str += "#$ -V\n"
    qsub_str += "#$ -N " + setting + "\n"
    qsub_str += "#$ -e " + options.language + "/output/" + setting + ".e -o " + options.language + "/output/" + setting + ".o\n"
    qsub_str += "echo 'Learning...'\n"
    qsub_str += "rm " + options.language + "/train_test/" + setting + "-best.ff\n"
    qsub_str += "java -Xmx20G -cp erma-src.jar driver.Learner -config=config/NER.cfg -features=" + options.language + "/train_test/" + setting + ".template -data=" + options.language + "/train_test/" + setting + ".train -out_ff=" + options.language + "/train_test/" + setting + "\n"
    if pipeline_sent:
        qsub_str += "echo 'Classifying...'\n"
        qsub_str += "java -Xmx20G -cp erma-src.jar driver.Classifier -config=config/NER.cfg -data=" + options.language + "/train_test/" + setting + ".new.test -features=" + options.language + "/train_test/" + setting + "-best.ff -pred_fname=" + options.language + "/train_test/" + setting + ".predictions"
    else:
        qsub_str += "echo 'Classifying...'\n"
        qsub_str += "java -Xmx20G -cp erma-src.jar driver.Classifier -config=config/NER.cfg -data=" + options.language + "/train_test/" + setting + ".test -features=" + options.language + "/train_test/" + setting + "-best.ff -pred_fname=" + options.language + "/train_test/" + setting + ".predictions"
    qsub.write(qsub_str)
    

if note != "":
    note = "." + note
if linear:
    setting = "linear" + note
elif joint:
    setting = "joint" + note
elif pipeline_NE:
    setting = "pipe_NE" + note
elif pipeline_sent:
    setting = "pipe_sent" + note
else:
    sys.stderr.write("What model am I doing?\n")
    sys.exit()

bad_words = read_known_words(bad_words_file)
good_words = read_known_words(good_words_file)
curse_words = read_known_words(curse_words_file)
prepositions = read_known_words(prepositions_file)
determiners = read_known_words(determiners_file)
syllable_structure = read_syllables_file(syllables_file)
for fid in other_feature_files:
    feature_name = fid.split(".txt")
    feature_name = feature_name[0]
    feature_file = open(fid, "r")
    open_feature_file = feature_file.readlines()
    feature_file.close()
    split_feature_name = feature_name.split("_")
    feature_type = split_feature_name[2]
    split_feature_name = feature_name.split("/")
    feature_name = split_feature_name[-1]
    extra_features[(feature_name, feature_type)] = read_known_words(open_feature_file)
tempt = open(options.language + "/train_test/" + setting + ".template", "w+")
traint = open(options.language + "/train_test/" + setting + ".train", "w+")
testt = open(options.language + "/train_test/" + setting + ".test", "w+")
qsub = open(options.language + "/qsub/" + setting + ".sh", "w+")

(train_features, train_skip_chains) = read_file(train)
train_hash = make_nodes(train_features)
make_all_features()
#sys.exit()
(test_features, test_skip_chains) = read_file(test)
test_hash = make_nodes(test_features, test=True)

if not just_test:
    if linear:
        print_linear_template(tempt)
    else:
        print_template(tempt)
    print_train_test(train_hash, traint, train_skip_chains)
print_train_test(test_hash, testt, test_skip_chains, True)

if pipeline_sent:
    testt.close()
    read_test = open(options.language + "/train_test/" + setting + ".test", "r").readlines()
    NE_predictions = open(pipeline_NE_predictions, "r").readlines()
    new_test = open(options.language + "/train_test/" + setting + ".new.test", "w+")
    # Writes out new test file.
    pipeline_class = pipe_class.Pipeline(read_test, NE_predictions, new_test)

if not just_test:
    print_qsub(qsub, setting)
