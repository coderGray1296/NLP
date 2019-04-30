#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import glob
import re
import codecs
import unicodedata
import getSignature as g
from argparse import ArgumentParser
from pipelineHelper import Pipeline as pipe
from collections import defaultdict


class ConllToErma:
    def __init__(self, args, bad_words, good_words, curse_words, prepositions, determiners, syllable_structure, other_features):
        self.just_test = args.just_test
        ## Model specifications
        self.model = args.model
        self.language = args.language
        ## Output note
        self.note = args.note
        ## Filtering features
        self.known_words = defaultdict(int)
        self.all_features = {}
        self.all_types = {}
        self.all_sent = {}
        self.punc = re.compile(u"[^A-Za-z0-9_ÁÉÍÓÚÜÑáéíóúüñ]", re.UNICODE)
        self.string_map = {".":"_P_" , ",":"_C_", "'":"_A_", "%":"_PCT_", "-":"_DASH_",
            "$":"_DOL_", "&":"_AMP_", ":":"_COL_", ";":"_SCOL_", "\\":"_BSL_", 
            "/":"_SL_", "`":"_QT_", "?":"_Q_", u"¿":"_QQ_", "=":"_EQ_", "*":"_ST_",
            "!":"_E_", u"¡":"_EE_", "#":"_HSH_", "@":"_AT_", "(":"_LBR_", ")":"_RBR_", 
            "\"":"_QT0_", u"Á":"_A_ACNT_", u"É":"_E_ACNT_", u"Í":"_I_ACNT_",
            u"Ó":"_O_ACNT_", u"Ú":"_U_ACNT_", u"Ü":"_U_ACNT0_", u"Ñ":"_N_ACNT_",
            u"á":"_a_ACNT_", u"é":"_e_ACNT_", u"í":"_i_ACNT_", u"ó":"_o_ACNT_",
            u"ú":"_u_ACNT_", u"ü":"_u_ACNT0_", u"ñ":"_n_ACNT_", u"º":"_deg_ACNT_"}
        ## Linguistic features
        self.other_features = other_features
        self.bad_words = bad_words
        self.good_words = good_words
        self.curse_words = curse_words  
        self.prepositions = prepositions
        self.determiners = determiners
        self.is_upper = re.compile("[A-ZÁÉÍÓÚÜÑ]", re.UNICODE)
        self.vowels = ''.join(syllable_structure[1])
        self.onsets = syllable_structure[0]
        self.codas = syllable_structure[2]
        self.negation = args.negation
        self.stops = ("#", ".", ",", "?", ":", ";", "-", "!")
        # Some other crap
        self.horizon_features = {}
        self.unique_id = args.unique_id
        self.clean_unique_id = args.unique_id
        # Factor between NEs.
        self.NE_links = args.NE_links
        # Factor between sent vars
        self.sent_links = args.sent_links
        self.with_sent = args.with_sent
        self.volitional = args.volitional
        self.lex_id = args.lex_id
        self.hidden_sent = args.hidden_sent
        self.no_cursing = args.no_cursing
        self.no_sentiment = args.no_sentiment
        self.no_polarity = args.no_polarity
        self.do_print_qsub = args.do_print_qsub
        self.reg_beta = args.reg_beta
        self.reg_func = args.reg_func
        self.pipeline_NEs = args.pipeline_NEs
        self.pipeline_NE_format = args.pipeline_NE_format
        self.feature_cutoff = args.feature_cutoff
        self.adjust_model_params(args)
        self.ngram_length = args.ngram_length
        if self.note != "":
            self.setting = self.model + "." + self.note
        else:
            self.setting = self.model
        
        
    def adjust_model_params(self, args):
        """ Sets necessary parameters for the given models. """
        if self.model == "linear":
            self.with_sent = False
            if args.with_sent:
                sys.stderr.write("Setting with_sent param to False ")
                sys.stderr.write("for linear model (by definition).\n")
        elif self.model == "joint":
            self.hidden_sent = True
            self.sent_links = True
            if not args.hidden_sent or not args.sent_links:
                sys.stderr.write("Setting hidden_sent and sent_links params ")
                sys.stderr.write("to True for joint model (by definition.)\n")
        elif self.model == "pipeline_NE":
            self.with_sent = False
            if args.with_sent:
                sys.stderr.write("Setting with_sent param to False for ")
                sys.stderr.write("first stage of pipeline model (by definition.)\n")
        # If we have an NE predictions file we're sticking in,
        # we're on the second stage of the pipeline
        elif self.model == "pipeline_sent":
            self.hidden_sent = True
            self.NE_links = False
            if args.NE_links:
                sys.stderr.write("Setting NE_links param to False for ")
                sys.stderr.write("second stage of pipeline model (by definition.)\n")
            if not self.pipeline_NEs:
                sys.stderr.write("Need output predictions from NE stage of the pipeline. \n") 
                sys.stderr.write("Use option --pipeline-NE-in=file and --pipeline-NE-format=erma for ERMA-formatted predictions.\n")
                sys.stderr.write("And option --pipeline-NE-in=file and --pipeline-NE-format=conll for CoNLL-formatted predictions.\n")
                sys.exit()
        self.note = args.note
        self.clean_unique_id = self.clean(self.unique_id)
        if self.clean_unique_id != self.unique_id:
            sys.stderr.write("Warning: " + self.unique_id + " contains illegal chars.\n")
            sys.stderr.write("Will print this out as " + self.clean_unique_id + ".\n")


    def escape(self, s):
        if type(s) is str:
            s = s.decode('utf-8')
        for val, repl in self.string_map.iteritems():
            s = s.replace(val,repl)
        return s


    def norm_digits(self, s):
        return re.sub("\d+","0",s, re.UNICODE)


    def clean(self, s):
        s = self.norm_digits(s.lower())
        s = self.escape(s)
        return self.punc.sub("_",s, re.UNICODE)


    def make_links(self, tweet_words):
        s_chains = defaultdict(dict)
        for (word, n) in tweet_words:
            # If it's capitalized...
            if self.is_upper.search(word[0]):
                possible_NE = word.lower()
                for (other_word, other_n) in tweet_words:
                    if other_n == n:
                        continue
                    if other_word.lower() == possible_NE and (other_n, n) not in s_chains[possible_NE]:
                        s_chains[possible_NE][(n, other_n)] = {}
        return s_chains


    def read_conll_file(self, in_file):
        tweet_words = {}
        skip_chains = {}
        word_hash = {}
        tweet_id = 0
        num_lines = 0
        for line in in_file:
            num_lines += 1
            if num_lines % 1000 == 0:
                sys.stdout.write(str(num_lines) + " lines processed.\n")
            line = line.strip()
            if line == "":
                continue
            elif line.startswith(self.unique_id):
                # Grab the new set of features and RVs.
                split_line = line.split()
                skip_chains[tweet_id] = self.make_links(tweet_words)
                tweet_id += 1
                word_hash[tweet_id] = {}
                tweet_words= {}
                n = 0
            else:
                split_line = line.split("\t")
                if len(split_line) != 8:
                    sys.stderr.write(str(tweet_id) + "\n")
                    sys.stderr.write(line + "\n")
                    sys.stderr.write("Weird line, formatted incorrectly.  Skipping.\n")
                    continue
                word = split_line[0].strip()
                # This is the NE.
                NE = split_line[1].strip()
                # Linear should never be specified in this case anyway.
                if self.volitional and not self.model == "linear": 
                    if NE.startswith("B") or NE.startswith("I"):
                        if not NE.endswith("ORGANIZATION") and not NE.endswith("PERSON"):
                            NE = "O"
                        else:
                            NE = re.sub("-\S+", "-VOLITIONAL", NE)
                NE = re.sub("-", "_", NE)
                brown_cluster5 = "_brown5_" + split_line[2].strip()
                # Jerboa features
                jerboa = "_jerboa_" + self.clean(split_line[3].strip())
                brown_cluster3 = "_brown3_" + split_line[4].strip()
                if len(split_line) > 5:
                    # This is the sentiment column.
                    prior_polarity = "_sent_" + self.clean(split_line[5].strip())
                    prior_ther_polarity = "_sent_ther_" + re.sub(":", "_", split_line[6].strip()).lower()
                    # This is the sentiment towards the entity.
                    targeted_polarity = split_line[-1].strip()
                    if self.no_polarity and (targeted_polarity == "positive" or targeted_polarity == "negative"):
                        targeted_polarity = "sentiment"
                    if self.no_sentiment and targeted_polarity != "_":
                        targeted_polarity = "volitional"
                else:
                    prior_polarity = "_"
                    theresa_polarity = "_"
                    targeted_polarity = "_"
                try:
                    esc_word = unicode(word, 'unicode-escape')
                except UnicodeDecodeError:
                    try:
                        sys.stderr.write("Could not decode " + word + "\n")
                        esc_word = unicode(word)
                        sys.stderr.write("Changing to: " + esc_word + "\n")
                    except UnicodeDecodeError:
                        sys.stderr.write("Stripping unicode for word " + word + ".\n")
                        esc_word = unicodedata.normalize('NFKD', unicode(word, 'utf-8')).encode('ASCII','ignore')
                word_hash[tweet_id][n] = (esc_word, NE, brown_cluster5, jerboa, brown_cluster3, prior_polarity, prior_ther_polarity, targeted_polarity)
                self.known_words[esc_word] += 1
                tweet_words[(esc_word, n)] = {}
                n += 1
        skip_chains[tweet_id] = self.make_links(tweet_words)
        return (word_hash, skip_chains)


    def bin_wl(self, length, feature_tmp):
        if length < 5:
            group = 1
        elif length < 8:
            group = 2
        else:
            group = 3
        feature = "_word_length_bin_" + str(group)
        feature_tmp[feature] = {}
        

    def bin_pos(self, pos, sent_len, feature_tmp):
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
        feature_tmp[feature] = {}
        
        
    def bin_msg(self, sent_len, feature_tmp):
        if sent_len <= 5:
            group = 1
        elif sent_len <= 10:
            group = 2
        else:
            group = 3
        feature = "_message_length_bin_" + str(group)
        feature_tmp[feature] = {}
        

    def syllabize(self, word):
        word = word.lower()
        split_word = word.split("-")
        sylls = re.findall("([^" + self.vowels + "]*?[" + self.vowels + "])", word, re.UNICODE) 
        sylls_tmp = re.findall("[^" + self.vowels + "]*?$", word, re.UNICODE)
        if sylls !=[] and sylls_tmp != []:
            sylls[-1] += sylls_tmp[0]
        return sylls


    def get_sonority(self, word):
        syllables = []
        #if not re.search("[A-Za-z]", word):
        #    return None
        split_word = word.split("-")
        for word in split_word:
            if word == "":
                continue
            syllables_tmp = self.syllabize(word)
            if syllables_tmp == []:
                #print "can't syllabize "
                #print word
                return "_cannot_syllabize"
            syllables += syllables_tmp
        last_pattern = None
        last_coda = ""
        for syll in syllables:
            split_syllable = re.findall("([^" + self.vowels + "]*)([" + self.vowels + "])([^" + self.vowels + "]*)", syll, re.UNICODE)
            split_syllable = split_syllable[0]
            onset = split_syllable[0]
            coda = split_syllable[-1]
            if (onset == "" or onset in self.onsets) and (coda == "" or coda in self.codas):
                last_coda = coda
                continue
            else:
                while onset not in self.onsets or coda not in self.codas or last_coda not in self.codas:
                    try:
                        last_coda += onset[0]
                        onset = onset[1:]
                    except IndexError:
                        return "_cannot_syllabize"
                last_coda = coda
        return None


    def set_pos_feature(self, pos, sent_len, feature_tmp):
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


    def set_stop_features(self, feature_tmp, prev_word, next_word):
        if prev_word[0] in self.stops:
            feature_tmp["_prev_in_stops"] = {}
        if next_word[0] in self.stops:
            feature_tmp["_next_in_stops"] = {}
            
            
    def set_extraling_features(self, feature_tmp, jerboa, word, lower_word):
        """ Relatively 'Extra-linguistic' features, like emoticons and laughs:  
            But biased for the languages we're developing in (Spanish and English), 
            and assuming their alphabet.  """
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
        if "haha" in lower_word or "jaja" in lower_word or "jeje" in lower_word or "hehe" in lower_word or "hihi" in lower_word or "jiji" in lower_word:
            feature_tmp["_sent_has_laugh"] = {}
    
    
    def set_ling_features(self, feature_tmp, lower_word):
        if re.search("^" + self.negation + "{2,}$", lower_word):
            feature_tmp["_sent_has_noo"] = {}
        if lower_word == self.negation:
            feature_tmp["_sent_has_no"] = {}
        if lower_word in self.bad_words:
            feature_tmp["_sent_has_mal"] = {}
        if lower_word in self.good_words: 
            feature_tmp["_sent_has_bien"] = {}
        if lower_word == "my" or lower_word == "mis" or lower_word == "mi":
            feature_tmp["_sent_has_mi"] = {}
        if lower_word == "nunca" or lower_word == "nadie":
            feature_tmp["_sent_has_no_word"] = {}
        if lower_word in self.curse_words and not self.no_cursing:
            feature_tmp["_sent_has_curse_word"] = {}
            
            
    def set_other_features(self, feature_tmp, lower_word):
        for (feature_file, feature_type) in self.other_features:
            if feature_type == "Suff":
                for suff in self.other_features[(feature_file, feature_type)]:
                    if lower_word.endswith(suff):
                        feature_tmp["_sent_" + feature_file] = {}
            elif feature_type == "Prefix":
                for pref in self.other_features[(feature_file, feature_type)]:
                    if lower_word.startswith(pref):
                        feature_tmp["_sent_" + feature_file] = {}
            if lower_word in self.other_features[(feature_file, feature_type)]:
                feature_tmp["_sent_" + feature_file] = {}
                
    
    def combine_features(self, feature_tmp):
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

    def escape_word(self, word):
        try:
            esc_word = unicode(word, 'unicode-escape')
        except UnicodeDecodeError:
            try:
                esc_word = unicode(word)
            except UnicodeDecodeError:
                esc_word = unicodedata.normalize('NFKD', unicode(word, 'utf8')).encode('ASCII','ignore')
        except TypeError:
            esc_word = word
        return esc_word


    def get_features(self, node, pos, sent_len, four_gram):
        """ Node: (word, NE, brown_cluster5, jerboa, brown_cluster3, prior_polarity, theresa_polarity, targeted_polarity) """
        feature_tmp = {}
        (word, NE, brown_cluster5, jerboa, brown_cluster3, prior_polarity, theresa_polarity, targeted_polarity) = node
        esc_word = self.escape_word(word)
        lower_word = esc_word.lower()
        normalized_word = self.clean(esc_word)
        prev_prev_word = four_gram[0]
        prev_word = four_gram[1]
        next_word = four_gram[-1]
        self.set_stop_features(feature_tmp, prev_word, next_word)
        # General features related to sentence position and length.
        self.set_pos_feature(pos, sent_len, feature_tmp)
        self.bin_pos(pos, sent_len, feature_tmp)
        self.bin_msg(sent_len, feature_tmp)
        self.bin_wl(len(word), feature_tmp)
        # Brown clusters
        feature_tmp[brown_cluster5] = {}
        feature_tmp[brown_cluster3] = {}
        # Jerboa features.
        feature_tmp[jerboa] = {}
        # Previous known sentiment
        # of this word
        feature_tmp[prior_polarity] = {}
        feature_tmp[theresa_polarity] = {}
        if prior_polarity[-1] != "_":
            feature_tmp["_is_sent"] = {}
        if theresa_polarity[-1] != "_": 
            feature_tmp["_is_sent_ther"] = {}
        self.set_extraling_features(feature_tmp, jerboa, word, lower_word)
        self.set_ling_features(feature_tmp, lower_word)
        self.set_other_features(feature_tmp, lower_word)
        # This is the label we're predicting.  MM:  If not linear?
        self.all_types[NE] = {}
        # This is the sentiment we're predicting.
        self.all_sent[targeted_polarity] = {}
        for feature in feature_tmp:
            if "sent_" in feature:
                self.horizon_features[feature] = {}
        # Specific lexical features for the word.
        if (len(esc_word) == 3 or len(esc_word) == 4) and esc_word not in self.determiners:
            feature_tmp["_is_tli"] = {}
        if jerboa == "_jerboa_none":
            syll_feature = self.get_sonority(lower_word)
            if syll_feature:
                feature_tmp[syll_feature] = {}
        if esc_word not in self.known_words or self.known_words[esc_word] < 3:
            if self.lex_id:
                oov = "_" + self.clean(g.getSignature(esc_word, language=self.language).unk)
                oov = unicodedata.normalize('NFKD', oov).encode('ASCII','ignore')
                feature_tmp[oov] = {}
        else:
            if self.lex_id:
                word_id = "_id_" + self.clean(esc_word)
                word_id = unicodedata.normalize('NFKD', word_id).encode('ASCII','ignore')
                feature_tmp[word_id] = {}
        simple_features = g.getSignature(esc_word, language=self.language).simple_unk_features
        for s in simple_features:
            feature_tmp[s] = {}
        if lower_word in self.determiners:
            feature_tmp["_is_determiner"] = {}
        if lower_word in self.prepositions:
            feature_tmp["_is_preposition"] = {}
        self.combine_features(feature_tmp)
        return feature_tmp


    def get_features_with_prefix(self, features_tweet_RV, tmp_features, prefix, test):
        for feature in tmp_features:
            feature = prefix + feature
            if not test:
                self.all_features[feature] = self.all_features.setdefault(feature,0) + 1
                features_tweet_RV[feature] = {}
            elif feature in self.all_features:
                features_tweet_RV[feature] = {}
        return features_tweet_RV


    def bin_sent(self, num_sent):
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


    def get_horizon_polarity(self, features_tweet_RV, features_tmp, n, test):
        i = 0
        while i < n - 3:
            for horizon_feature in self.horizon_features:
                if horizon_feature in features_tmp[i]:
                    feature = "_prev" + horizon_feature
                    if not test:
                        features_tweet_RV[feature] = {}
                        self.all_features[feature] = self.all_features.setdefault(feature,0) + 1
                    elif feature in self.all_features:
                        features_tweet_RV[feature] = {}
            i += 1
        if n < 3:
            i = 0
        else:
            i = n - 3
        while i < n:                                
            for horizon_feature in self.horizon_features:                    
                if horizon_feature in features_tmp[i]:
                    feature = "_immediate_prev" + horizon_feature
                    if not test:
                        features_tweet_RV[feature] = {}
                        self.all_features[feature] = self.all_features.setdefault(feature,0) + 1
                    elif feature in self.all_features:
                        features_tweet_RV[feature] = {}
            i += 1
        i = n + 3
        while i < len(features_tmp):
            for horizon_feature in self.horizon_features:
                if horizon_feature in features_tmp[i]:
                    feature = "_next" + horizon_feature
                    if not test:
                        features_tweet_RV[feature] = {}
                        self.all_features[feature] = self.all_features.setdefault(feature,0) + 1
                    elif feature in self.all_features:
                        features_tweet_RV[feature] = {}
            i += 1
        i = n
        while i < n + 3 and i < len(features_tmp):
            for horizon_feature in self.horizon_features:
                if horizon_feature in features_tmp[i]:
                    feature = "_immediate_next" + horizon_feature
                    if not test:
                        features_tweet_RV[feature] = {}
                        self.all_features[feature] = self.all_features.setdefault(feature,0) + 1
                    elif feature in self.all_features:
                        features_tweet_RV[feature] = {}
            i += 1
        num_sent = 0
        for n in features_tmp:
            features = features_tmp[n]
            if "_is_sent" in features:
                num_sent += 1
        overall_sent_features = self.bin_sent(num_sent)
        for overall_sent_feature in overall_sent_features:
            feature = "_sent_overall" + overall_sent_feature
            if not test:
                features_tweet_RV[feature] = {}
                self.all_features[feature] = {}
            elif feature in self.all_features:
                features_tweet_RV[feature] = {}
        return features_tweet_RV


    def get_four_gram(self, tmp_hash, n):
        try:
            prev_word = tmp_hash[n-1][0]
        except KeyError:
            prev_word = "SENT_START"
        try:
            prev_prev_word = tmp_hash[n-2][0]
        except KeyError:
            prev_prev_word = "SENT_START"
        try:
            next_word = tmp_hash[n+1][0]
        except KeyError:
            next_word = "SENT_END"
        word = tmp_hash[n][0]
        return (prev_prev_word, prev_word, word, next_word)


    def make_nodes(self, word_hash, test=False):
        features = {}
        has_rel_NE = {}
        for tweet_id in word_hash:
            features[tweet_id] = {}
            features_tmp = {}
            n = 0
            while n < len(word_hash[tweet_id]):
                # (word, NE, jerboa, prior_polarity, prior_ther_polarity, targeted_polarity) = node
                node = word_hash[tweet_id][n]
                NE = node[1]
                # Don't include example we don't need during training.
                # Should be done earlier, but just in case.
                if not test:
                    if ("ORGANIZATION" in NE or "PERSON" in NE or "VOLITIONAL" in NE):
                        has_rel_NE[tweet_id] = {}
                else:
                    has_rel_NE[tweet_id] = {}
                four_gram = self.get_four_gram(word_hash[tweet_id], n)
                features_tmp[n] = self.get_features(node, n, len(word_hash[tweet_id]), four_gram)
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
                # Do not encode features for sentiment we do not know about
                # If we're only predicting sentiment
                if (self.model == "pipeline_sent" and sent == "_"):
                    n += 1
                    continue
                prefix = "word"
                features[tweet_id][(RV, label, sent)] = self.get_features_with_prefix({}, features_tmp[n], prefix, test)
                p = 0
                while p < self.ngram_length and n+p < len(word_hash[tweet_id]) - 1:
                    p += 1
                    ahead_prefix = "wordp" + str(p)
                    features[tweet_id][(RV, label, sent)] = self.get_features_with_prefix(features[tweet_id][(RV, label, sent)], features_tmp[n+p], ahead_prefix, test)
                p = 0
                while p < self.ngram_length and n-p > 0:
                    p += 1
                    behind_prefix = "wordm" + str(p)
                    features[tweet_id][(RV, label, sent)] = self.get_features_with_prefix(features[tweet_id][(RV, label, sent)], features_tmp[n-p], behind_prefix, test)
                features[tweet_id][(RV, label, sent)] = self.get_horizon_polarity(features[tweet_id][(RV, label, sent)], features_tmp, n, test)
                n += 1
        return features


    def cutoff_features(self):
        keys = self.all_features.keys()
        for feature in keys:
            if self.all_features[feature] < self.feature_cutoff:
                del self.all_features[feature]
                
    
    def print_variable(tt, var_type, var_id, label, var_status="pred"):
        if var_status == "pred":
            print >> tt, var_type + " " + var_id + "=" + label + ";"
        elif var_status == "latent":
            print >> tt, var_type + " " + var_id + ";"
        elif var_status == "observed":
            print >> tt, var_type + " " + var_id + "=" + label + " in;"

        
    def print_linear_variables(self, tt, RV, label, sent):
        if sent == "_":
            # Treat it as latent.
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

    def print_joint_variables(self, tt, RV, label, sent, sent_chains):
        # We don't know the label here; treat as latent.
        # (Shouldn't really happen if we've annotated all the data)
        if label == "_":
            print >> tt, "NE " + RV + ";"
        else:
            print >> tt, "NE " + RV + "=" + label + ";"
        SV = "S" + RV[1:]
        # We don't know the sentiment label here; treat as latent.
        if sent == "_":
            print >> tt, "SENT " + SV + ";"
        else:
            print >> tt, "SENT " + SV + "=" + sent + ";"
        sent_chains[int(SV[1:])] = {}
        return sent_chains

    def print_pipe_NE_variables(self, tt, RV, label):
        # We don't know the label here; treat as latent.
        # (Shouldn't really happen if we've annotated all the data)
        if label == "_":
            print >> tt, "NE " + RV + ";"
        else:
            print >> tt, "NE " + RV + "=" + label + ";"

    def print_pipe_sent_variables(self, tt, RV, pred_label, sent, sent_chains):
        print >> tt, "NE " + RV + "=" + pred_label + " in;"
        SV = "S" + RV[1:]
        # We don't know the sent label here; treat as latent.
        if sent == "_" and self.hidden_sent:
            print >> tt, "SENT " + SV + ";"
            sent_chains[int(SV[1:])] = {}
        else:
            print >> tt, "SENT " + SV + "=" + sent + ";"
            sent_chains[int(SV[1:])] = {}
        return sent_chains
        

    def print_skip_chain(self, tt, skip_chains):
        for chain in skip_chains:
            for skip_chain in skip_chains[chain]:
                n = skip_chain[0]
                n2 = skip_chain[1]
                RV1 = "W" + str(n)
                RV2 = "W" + str(n2)
                print >> tt, "skip_chain(" + RV1 + "," + RV2 + ");"


    def print_sent_chain(self, tt, pipe_sent_vars):
        sent_vars = pipe_sent_vars.keys()
        sent_vars.sort()
        s = 0
        while s + 1 < len(sent_vars):
            s_num = sent_vars[s]
            s_num2 = sent_vars[s+1]
            print >> tt, "sent_chain(S" + str(s_num) + ",S" + str(s_num2) + ");"
            s += 1


    def print_linear_features(self, tt, RV, label, sent, in_hash):
        for feature in sorted(in_hash[(RV, label, sent)]):
            if feature in self.all_features:
                print >> tt, feature + "(" + RV + ");"
        # BIAS feature.
        #print >> tt, "ne_bias(" + RV + ");"
        last_RV_num = int(RV.strip("W")) - 1
        if last_RV_num < 0:
            last_RV = "BEGIN"
        else:
            last_RV = "W" + str(last_RV_num)
        # BIAS feature.
        print >> tt, "link(" + last_RV + "," + RV + ");"


    def print_joint_features(self, tt, RV, label, sent, in_hash):
        SV = "S" + RV[1:]
        for feature in sorted(in_hash[(RV, label, sent)]):
            if feature in self.all_features:
                print >> tt, feature + "(" + RV + ");"
                if "_sent_" in feature or "_id_" in feature:
                    print >> tt, feature + "_link(" + RV + "," + SV + ");"
                    print >> tt, feature + "_sent(" + SV + ");"
        # BIAS features.
        #print >> tt, "ne_bias(" + RV + ");"
        #print >> tt, "sent_bias(" + SV + ");"
        print >> tt, "sent_link(" + RV + "," + SV + ");"
        last_RV_num = int(RV.strip("W")) - 1
        if last_RV_num < 0:
            last_RV = "BEGIN"
        else:
            last_RV = "W" + str(last_RV_num)
        # BIAS feature.
        print >> tt, "link(" + last_RV + "," + RV + ");"


    def print_pipe_NE_features(self, tt, RV, label, sent, in_hash):
        for feature in sorted(in_hash[(RV, label, sent)]):
            if feature in self.all_features:
                print >> tt, feature + "(" + RV + ");"
        # BIAS feature.
        #print >> tt, "ne_bias(" + RV + ");"
        last_RV_num = int(RV.strip("W")) - 1
        if last_RV_num < 0:
            last_RV = "BEGIN"
        else:
            last_RV = "W" + str(last_RV_num)
        # BIAS feature.
        print >> tt, "link(" + last_RV + "," + RV + ");"


    def print_pipe_sent_features(self, tt, RV, label, sent, in_hash, sent_chains):
        Snum = int(RV[1:])
        if Snum not in sent_chains:
            return
        SV = "S" + str(Snum)
        for feature in sorted(in_hash[(RV, label, sent)]):
            if feature in self.all_features:
                if "_sent_" in feature or "_id_" in feature:
                    print >> tt, feature + "_link(" + RV + "," + SV + ");"
                    print >> tt, feature + "_sent(" + SV + ");"
        # BIAS features.
        #print >> tt, "sent_bias(" + SV + ");"
        print >> tt, "sent_link(" + RV + "," + SV + ");"


    def print_train_test(self, in_hash, tt, skip_chains, test=False):
        for tweet_id in sorted(in_hash):
            if not in_hash[tweet_id]:
                continue
            sent_chains = {}
            print >> tt, "//" + self.clean_unique_id + " " + str(tweet_id)
            print >> tt, "example:"
            for (RV, label, sent) in sorted(in_hash[tweet_id]):
                if self.model == "linear":
                    self.print_linear_variables(tt, RV, label, sent)
                elif self.model == "joint":
                    sent_chains = self.print_joint_variables(tt, RV, label, sent, sent_chains)
                elif self.model == "pipeline_NE":
                    self.print_pipe_NE_variables(tt, RV, label)
                elif self.model == "pipeline_sent":
                    sent_chains = self.print_pipe_sent_variables(tt, RV, label, sent, sent_chains)
            print >> tt, "features:"
            if self.sent_links:
                self.print_sent_chain(tt, sent_chains)
            if not self.model == "pipeline_sent":
                self.print_skip_chain(tt, skip_chains[tweet_id])
            for (RV, label, sent) in sorted(in_hash[tweet_id]):
                if self.model == "linear":
                    self.print_linear_features(tt, RV, label, sent, in_hash[tweet_id])
                elif self.model == "joint":
                    self.print_joint_features(tt, RV, label, sent, in_hash[tweet_id])
                elif self.model == "pipeline_NE":
                    self.print_pipe_NE_features(tt, RV, label, sent, in_hash[tweet_id])
                elif self.model == "pipeline_sent":
                    self.print_pipe_sent_features(tt, RV, label, sent, in_hash[tweet_id], sent_chains)

    def print_linear_template(self, tt):
        print >> tt, "types:"
        if self.no_sentiment: # MM: Change to self.all_types join?
            print >> tt, "NESENT:=[Bvolitional,I,O]"
        elif self.no_polarity:
            print >> tt, "NESENT:=[Bneutral,Bsentiment,I,Isentiment,Ineutral,O]"
        else:
            print >> tt, "NESENT:=[Bneutral,Bpositive,Bnegative,I,Ineutral,Ipositive,Inegative,O]" 
        print >> tt, ""
        print >> tt, "features:"
        if self.NE_links:
            print >> tt, "link(NESENT,NESENT):=[*,*]"
            print >> tt, "skip_chain(NESENT,NESENT):=[*,*]"
        #print >> tt, "ne_bias(NESENT):=[*]"
        for feat in self.all_features:
            print >> tt, feat + "(NESENT):=[*]"

                    
    def print_template(self, tt):
        print >> tt, "types:"
        print >> tt, "NE:=[" + ",".join(self.all_types) + "]" 
        if self.with_sent:  # MM: Change to self.all_sent join?
            if self.no_polarity:
                print >> tt, "SENT:=[sentiment,neutral]"
            else:
                print >> tt, "SENT:=[positive,negative,neutral]" 
        print >> tt, ""
        print >> tt, "features:"
        if self.model != "pipeline_sent":
            print >> tt, "skip_chain(NE,NE):=[*,*]"
        if self.sent_links:
            print >> tt, "sent_chain(SENT,SENT):=[*,*]"
        if self.NE_links:
            print >> tt, "link(NE,NE):=[*,*]"
        if self.with_sent:
            # True if joint or pipeline_sent
            print >> tt, "sent_link(NE,SENT):=[*,*]"
            #print >> tt, "sent_bias(SENT):=[*]"
            # True in joint model
            if self.sent_links:
                print >> tt, "sent_sent_link(SENT,SENT):=[*,*]"
        #print >> tt, "ne_bias(NE):=[*]"
        for feat in self.all_features:
            if self.model != "pipeline_sent":
                print >> tt, feat + "(NE):=[*]"
            if self.with_sent:
                # True if joint or pipeline_sent:
                if "_sent_" in feat or "_id_" in feat:
                    print >> tt, feat + "_link(NE,SENT):=[*,*]"
                    print >> tt, feat + "_sent(SENT):=[*]"

    def print_qsub(self, qsub):
        qsub_str = "#!/bin/bash\n\n"
        qsub_str += "#$ -S /bin/bash\n"
        qsub_str += "#$ -cwd\n"
        qsub_str += "#$ -l num_proc=1,h_rt=36:00:00,h_vmem=40g,mem_free=40g\n"
        qsub_str += "#$ -V\n"
        qsub_str += "#$ -N " + self.setting + "\n"
        qsub_str += "#$ -e " + self.language + "/output/" + self.setting + ".e\n"
        qsub_str += "#$ -o " + self.language + "/output/" + self.setting + ".o\n"
        if not self.just_test:
            qsub_str += "echo 'Learning...'\n"
            qsub_str += "rm " + self.language + "/train_test/" + self.setting + "-best.ff\n"
            qsub_str += "java -Xmx20G -cp erma-src.jar driver.Learner -config=config/NER.cfg"
            #qsub_str += " -reg_func=" + self.reg_func
            #qsub_str += " -reg_beta=" + self.reg_beta
            qsub_str += " -features=" + self.language + "/train_test/" + self.setting + ".template"
            qsub_str += " -data=" + self.language + "/train_test/" + self.setting + ".train"
            qsub_str += " -out_ff=" + self.language + "/train_test/" + self.setting
            qsub_str += "\n"
        qsub_str += "echo 'Classifying...'\n"
        qsub_str += "java -Xmx20G -cp erma-src.jar driver.Classifier -config=config/NER.cfg"
        if self.model == "pipeline_sent":
            qsub_str += " -data=" + self.language + "/train_test/" + self.setting + ".new.test" 
        else: 
            qsub_str += " -data=" + self.language + "/train_test/" + self.setting + ".test"
        qsub_str += " -features=" + self.language + "/train_test/" + self.setting + "-best.ff"
        qsub_str += " -pred_fname=" + self.language + "/train_test/" + self.setting + ".predictions"
        qsub.write(qsub_str)    

    def write_out(self, train_hash, train_skip_chains, test_hash, test_skip_chains):
        if self.model == "pipeline_sent":
            # This is just running the sentiment testing predictions.
            # We could also train a model that has NE predictions within
            # the training data.
            NE_in_file = codecs.open(self.pipeline_NEs, "r", "utf-8").readlines()
            p = pipe()
            NE_predictions = p.read_NE_predictions(NE_in_file)
            # Sets up to write out new test file with observed NEs; 
            # in test, this is the output of the last stage of the pipeline.
            # However, we can still evaluate isolated sentiment predictions just over 
            # *known (gold) volitional NE spans*; we do not know sentiment of unknown volitional spans.
            test_hash = p.combine_NEs_with_sent(test_hash, NE_predictions)
            testt = codecs.open(self.language + "/train_test/" + self.setting + ".new.test", "w+", "utf-8")
        else:
            testt = codecs.open(self.language + "/train_test/" + self.setting + ".test", "w+", "utf-8")
        self.print_train_test(test_hash, testt, test_skip_chains, True)
        tempt = codecs.open(self.language + "/train_test/" + self.setting + ".template", "w+", "utf-8")
        if self.model == "linear":
            self.print_linear_template(tempt)
        else:
            self.print_template(tempt)
        if not self.just_test:
            traint = codecs.open(self.language + "/train_test/" + self.setting + ".train", "w+", "utf-8")
            self.print_train_test(train_hash, traint, train_skip_chains)
        if self.do_print_qsub:
            qsub = codecs.open(self.language + "/qsub/" + self.setting + ".sh", "w+", "utf-8")
            self.print_qsub(qsub)
                



def read_known_words(known_words_file):
    word_list = []
    for line in known_words_file:
        line = line.strip()
        if line == "":
            continue
        line = line.lower()
        word_list += [line]
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
    

def setArgumentParser():
    parser = ArgumentParser(usage='%(prog)s [options]')
    parser.add_argument("--train", help="Training file.")
    parser.add_argument("--test", help="Testing file.")
    parser.add_argument("--model", dest="model", help="Specify model type (joint, linear, pipeline_NE, pipeline_sent).", default="joint")
    parser.add_argument("--unique-id", help="Unique identifier for each sentence.", default="## Tweet", dest="unique_id")
    parser.add_argument("--no-lex-id", dest="lex_id", help="Use lexical identity.", action="store_false", default=True)
    parser.add_argument("--sent-links", dest="sent_links", help="Specify whether to include links between sentiment variables.", 
                        action="store_true", default=False)
    parser.add_argument("--no-NE-links", dest="NE_links", help="Specify not to include links between NE variables.", 
                        action="store_false", default=True)
    parser.add_argument("--hidden-sent", dest="hidden_sent", help="Specify whether to include sentiment as latent variables in cases without a sentiment assignment.", action="store_true", default=False)
    parser.add_argument("--no-sent", dest="with_sent", help="Specify whether to include sentiment.", action="store_false", default=True)
    parser.add_argument("--feature-cutoff", dest="feature_cutoff", help="Limit at which to call a feature OOV.", action="store", type=int, default=2)
    parser.add_argument("--ngram-length", dest="ngram_length", help="How many words before/after current word to consider", action="store", type=int, default=3)
    parser.add_argument("--no-cursing", dest="no_cursing", help="Specifying whether to include curse words.", 
                        action="store_true", default=False)
    parser.add_argument("--no-polarity", dest="no_polarity", help="Collapse positive/negative into one category.", 
                        action="store_true", default=False)
    parser.add_argument("--negation", dest="negation", help="Language-specific negation marker", 
                            action="store", default="no")
    parser.add_argument("--no-sentiment", dest="no_sentiment", help="Just predict volitional entities", 
                        action="store_true", default=False)
    parser.add_argument("--volitional", dest="volitional", help="Collapse Person/Org into volitional", 
                        action="store_true", default=False)
    parser.add_argument("--just-test", dest="just_test", help="Just make a test file.", 
                        action="store_true", default=False)
    parser.add_argument("--note", dest="note", help="Note to append to files.", 
                        action="store", default="")
    parser.add_argument("--language", dest="language", help="Set the language ('es' or 'en').", 
                        action="store", default="es")
    parser.add_argument("--reg-beta", dest="reg_beta", help="Regularization strength.", 
                        action="store", default="0.0")
    parser.add_argument("--reg-func", dest="reg_func", help="Regularization Function (Zero, L1, L2)", 
                        action="store", default="L2")
    parser.add_argument("--no-print-qsub", dest="do_print_qsub", help="do not print out a qsub submit file", 
                        action="store_false", default=True)
    parser.add_argument("--pipeline-NE-in", dest="pipeline_NEs", help="in file with pipelined NEs", action="store", default=False)
    parser.add_argument("--pipeline-NE-format", dest="pipeline_NE_format", help="format for input from last stage of pipeline (erma or conll).", default="erma")

    return parser
    
    
def set_features_from_files(other_feature_files):
    other_features = {}
    """ Generic function to read in arbitrary features """
    for fid in other_feature_files:
        feature_name = fid.split(".txt")
        feature_name = feature_name[0]
        feature_file = codecs.open(fid, "r", "utf-8")
        open_feature_file = feature_file.readlines()
        feature_file.close()
        split_feature_name = feature_name.split("_")
        feature_type = split_feature_name[2]
        split_feature_name = feature_name.split("/")
        feature_name = split_feature_name[-1]
        other_features[(feature_name, feature_type)] = read_known_words(open_feature_file)
    return (other_features)

def main(args):
    """ Features used for all languages.  USE UTF-8 ENCODING.
        Here, I assume my directory setup....=) """
    bad_words_file = codecs.open(args.language + "/feature_files/bad_words", "r", "utf-8").readlines()
    bad_words = read_known_words(bad_words_file)
    
    good_words_file = codecs.open(args.language + "/feature_files/good_words", "r", "utf-8").readlines()
    good_words = read_known_words(good_words_file)

    curse_words_file = codecs.open(args.language + "/feature_files/curse_words", "r", "utf-8").readlines()
    curse_words = read_known_words(curse_words_file)

    prepositions_file = codecs.open(args.language + "/feature_files/prepositions", "r", "utf-8").readlines()
    prepositions = read_known_words(prepositions_file)

    determiners_file = codecs.open(args.language + "/feature_files/determiners", "r", "utf-8").readlines()
    determiners = read_known_words(determiners_file)

    syllables_file = codecs.open(args.language + "/feature_files/syllables", "r", "utf-8").readlines()
    syllable_structure = read_syllables_file(syllables_file)

    other_feature_files = glob.glob(args.language + "/feature_files/*.txt")
    other_features = set_features_from_files(other_feature_files)
        
    ermaObj = ConllToErma(args, bad_words, good_words, curse_words, prepositions, \
                determiners, syllable_structure, other_features)

    if not args.just_test:
        # Input training file.
        train_id = open(args.train, "r")
        train = train_id.readlines()
        train_id.close()
        sys.stdout.write("Reading training file...\n")
        (train_features, train_skip_chains) = ermaObj.read_conll_file(train)
        sys.stdout.write("Building model...\n")
        train_hash = ermaObj.make_nodes(train_features)
        # Freeze the known features based on what's seen in the training data
        ermaObj.cutoff_features()
    else:
        train_hash = {}
        train_skip_chains = {}
    # Input testing file.
    test_id = open(args.test, "r")
    test = test_id.readlines()
    test_id.close()
    sys.stdout.write("Reading test file...\n")
    (test_features, test_skip_chains) = ermaObj.read_conll_file(test)
    sys.stdout.write("Building model...\n")
    test_hash = ermaObj.make_nodes(test_features, test=True)
    ermaObj.write_out(train_hash, train_skip_chains, test_hash, test_skip_chains)
    
if __name__ == '__main__':
    # This will always be the case with my current usage scenario.
    # If it's not, "args" must be defined separately.
    parser = setArgumentParser()
    args = parser.parse_args()
    main(args)
