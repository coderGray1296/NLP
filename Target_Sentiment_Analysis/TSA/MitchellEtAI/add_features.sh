#!/bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -l num_proc=1,h_rt=1:00:00,h_vmem=120g,mem_free=120g
#$ -V

if [ -z "$NER_SENT_HOME" ];
then
    echo "NER_SENT_HOME variable unset; assuming current working directory."
    NER_SENT_HOME=`pwd`
fi
DIR=$NER_SENT_HOME

if [ $# -ne 2 ]; then
    echo "Usage: `basename $0` simple_conll_file language[en|es]"
    exit 65
fi
echo "Simple conll format file: " $1
echo "Language: " $2
language=$2
new_1=${1##*/}
#echo "Turker CSV file..." $2
#echo "Adding Turk annotations..."
#python ${DIR}/scripts/parse_hit-forErma.py ${2} ${1} ${DIR}/annotations/${1}.annot
echo "Getting features..."
cat ${1} | python ${DIR}/scripts/extend_sentiment.py > ${DIR}/${language}/annotations/${new_1}.annot.more_sent 
cat ${1} | awk '{print $1}' > ${DIR}/${language}/annotations/${new_1}.word
echo "Getting Brown clusters..."
cat ${DIR}/${language}/annotations/${new_1}.word | python ${DIR}/scripts/clusters_to_pos.py ${DIR}/${language}/feature_files/brown_clusters 5 > ${DIR}/${language}/annotations/${new_1}.clusters5
cat ${DIR}/${language}/annotations/${new_1}.word | python ${DIR}/scripts/clusters_to_pos.py ${DIR}/${language}/feature_files/brown_clusters 3 > ${DIR}/${language}/annotations/${new_1}.clusters3
echo "Getting Jerboa features..."
java -DTwitterTokenizer.unicode=jerboa/unicode.csv -DTwitterTokenizer.full=true -cp jerboa/jerboa.jar edu.jhu.jerboa.processing.TwitterTokenizer ${DIR}/${language}/annotations/${new_1}.word | python ${DIR}/scripts/jerboa_to_conll.py > ${DIR}/${language}/annotations/${new_1}.jerboa
echo "Getting Sentiment Lexicon features..."
cat ${DIR}/${language}/annotations/${new_1}.word | python ${DIR}/scripts/sentiment_to_conll.py --language=${language} > ${DIR}/${language}/annotations/${new_1}.sentiment
echo "Combining all the aforementioned stuff..."
python ${DIR}/scripts/combine.py ${DIR}/${language}/annotations/${new_1}.annot.more_sent ${DIR}/${language}/annotations/${new_1}.clusters5 ${DIR}/${language}/annotations/${new_1}.jerboa ${DIR}/${language}/annotations/${new_1}.clusters3 ${DIR}/${language}/annotations/${new_1}.sentiment > ${DIR}/${language}/annotations/${new_1}.conll.train_test

cat ${DIR}/${language}/annotations/${new_1}.conll.train_test | python ${DIR}/scripts/check_columns.py

