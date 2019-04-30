import sys
import re
import codecs
import unicodedata


class Sentiment:
    def __init__(self, langa):
        if langa == "es":
            self.prev_polarities = codecs.open("es/SpanishSentimentLexicons/fullStrengthLexicon.txt", "r", 'utf-8').readlines()
            self.prev_medium_polarities = codecs.open("es/SpanishSentimentLexicons/mediumStrengthLexicon.txt", "r", 'utf-8').readlines()
            self.prev_ther_polarities = codecs.open("es/SpanishSentimentLexicons/spanishSubjLex-v1.1.tsv", "r", 'utf-8').readlines()
        elif langa == "en":
            self.prev_ther_polarities = codecs.open("en/EnglishSentimentLexicons/englishSubjLex-MPQA.tsv", "r", "utf-8").readlines()
            self.prev_polarities = codecs.open("en/EnglishSentimentLexicons/SentiWordNet_3.0.0_20130122.txt", "r", "utf-8").readlines()
        else:
            sys.stderr.write("Sorry, did not understand language specification -- please type 'english' or 'spanish'.\n")
        self.langa = langa
        self.previous_polarities = {}
        self.previous_medium_polarities = {}
        self.previous_ther_polarities = {"1":{}, "2":{}, "3":{}}
        self.read_theresa_polarities()
        if langa == "es":
            self.read_prev_polarities()
            self.read_prev_medium_polarities()
        else:
            self.read_senti_polarities()

    def calc_senti_sentiment(self, posScore, negScore):
        p = float(posScore)
        n = float(negScore)
        if p + n > .5:
            if p > n:
                return "positive"
            else:
                return "negative"
        else:
            return "neutral"

    def read_senti_polarities(self):
        for line in self.prev_polarities:
            if line.strip().startswith("#"):
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
                        sentiment = self.calc_senti_sentiment(posScore, negScore)
                        if sentiment == "neutral":
                            continue
                        if num == "1":
                            self.previous_polarities[word] = sentiment
                        else:
                            self.previous_medium_polarities[word] = sentiment
                    else:
                        break

        
    def read_theresa_polarities(self):
        for line in self.prev_ther_polarities:
            if line.strip().startswith("#"):
                continue
            line = line.strip()
            split_line = line.split()
            word = split_line[0].lower()
            polarity = split_line[1]
            score = float(split_line[2])
            if score <= .50:
                bin = "1"
            elif score <= .75:
                bin = "2"
            else:
                bin = "3"
            self.previous_ther_polarities[bin][word] = polarity

    def read_prev_polarities(self):
        for line in self.prev_polarities:
            line = line.strip()
            split_line = line.split()
            word = split_line[0].lower()
            polarity = split_line[2]
            if polarity == "neg":
                polarity = "negative"
            elif polarity == "pos":
                polarity = "positive"
            self.previous_polarities[word] = polarity

    def read_prev_medium_polarities(self):
        for line in self.prev_medium_polarities:
            line = line.strip()
            split_line = line.split()
            word = split_line[0].lower()
            polarity = split_line[2]
            if polarity == "neg":
                polarity = "negative"
            elif polarity == "pos":
                polarity = "positive"
            self.previous_medium_polarities[word] = polarity

    def get_ther_sentiment_polarity(self, word):
        word = word.lower()
        ther_sentiment_polarity = "_"
        if self.langa == "es" and word == "felicidades":
            ther_sentiment_polarity = "THER_SENT_3:positive"
            return ther_sentiment_polarity
        # Leave out 1.  And 2.
        for bin in ("3"):
            if word in self.previous_ther_polarities[bin]:
                prev_polarity = self.previous_ther_polarities[bin][word]
                ther_sentiment_polarity = "THER_SENT_" + bin + ":" + prev_polarity
        return ther_sentiment_polarity

    def get_sentiment_polarity(self, word):
        word = word.lower()
        stupid_word = unicodedata.normalize('NFKD', unicode(word, 'utf8')).encode('ASCII','ignore')
        if stupid_word in self.previous_polarities:
            prev_polarity = self.previous_polarities[stupid_word]
            sentiment_polarity = "SENT:" + prev_polarity
        #elif word == "felicidades":
            # Oops.  We accidentally said that this word was positive
            # at some point when making controls,
            # although it wasn't in the lexicon.
        #    sentiment_polarity = "SENT:positive"
        elif stupid_word in self.previous_medium_polarities:
            prev_medium_polarity = self.previous_medium_polarities[stupid_word]
            sentiment_polarity = "SENT_MED:" + prev_medium_polarity
        else:
            sentiment_polarity = "_"
        return sentiment_polarity
