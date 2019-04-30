#!/bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -l num_proc=1,h_rt=36:00:00,h_vmem=40g,mem_free=40g
#$ -V
#$ -N tokenize

java -DTwitterTokenizer.unicode=jerboa/unicode.csv -DTwitterTokenizer.full=false -cp jerboa/jerboa.jar edu.jhu.jerboa.processing.TwitterTokenizer English_tweets.txt > English_tweets.tokenized
