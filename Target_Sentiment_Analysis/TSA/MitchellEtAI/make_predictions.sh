#!/bin/bash


#new_1=${1##*/}
#./scripts/add_features.sh ${1}

python scripts/conll_to_erma-more_features.py annotations/train.conll ${1} --language=en --linear --just-test --noID

java -Xmx10G -cp erma-src.jar driver.Classifier -config=config/NER.cfg -data=linear.test -features=English.linear.model -pred_fname=English.linear.predictions

python scripts/label.py English.linear.predictions > English.linear.predictions.columns

python scripts/merge_predictions_with_annot.py ${1} English.linear.predictions.columns > ${1}.out
