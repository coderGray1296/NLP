#!/bin/bash
#$ -cwd
#$ -V
#$ -l h_rt=36:00:00

# Top 5 tweets:  Facebook, Twitter, Chavez, Carlos, Juan.  Total = 507
#grep -w "\S-ORGANIZATION\|\S-PERSON" PerOrgNEs-all.conll_simplified | awk '{print tolower($1) "\t" $2}' | sort | uniq -c | sort -nr | head -20

if [ $# -lt 2 ]; then
    echo "Usage: `basename ${0}` language command [other_options]"
    exit 65
fi

conditions=( with_lex_id with_lex_id.no_polarity )
#conditions=( with_lex_id.no_polarity )
language=$1
command=$2

train_test_dir=${language}/train_test

if [ "$command" == train_turk ]; then
    if [ $# -ne 4 ]; then
        echo "Usage: `basename ${0}` language train_turk turk_file turk_number"
        exit 65
    fi

    turk_file=$3
    turk_number=$4
    for num in 1; do cat ${language}/annotations/train.${num}.conll.train_test spacer ${turk_file} > ${language}/annotations/train.${num}+turk${turk_number}.conll.train_test; done

    for num in 1; do python scripts/ConllToErma.py --train=${language}/annotations/train.${num}+turk${turk_number}.conll.train_test --test=${language}/annotations/test.${num}.conll.train_test --model=linear --note=with_lex_id.${num}.volitional.turk.${turk_number} --language=${language}; done
    for num in 1; do python scripts/ConllToErma.py --train=${language}/annotations/train.${num}+turk${turk_number}.conll.train_test --test=${language}/annotations/test.${num}.conll.train_test --model=joint --volitional --note=with_lex_id.${num}.volitional.turk.${turk_number} --language=${language}; done
    for num in 1; do python scripts/ConllToErma.py --train=${language}/annotations/train.${num}+turk${turk_number}.conll.train_test --test=${language}/annotations/test.${num}.conll.train_test --model=pipeline_NE --volitional --note=with_lex_id.${num}.volitional.turk.${turk_number} --language=${language}; done

    for num in 1; do python scripts/ConllToErma.py --train=${language}/annotations/train.${num}+turk${turk_number}.conll.train_test --test=${language}/annotations/test.${num}.conll.train_test --model=linear --note=with_lex_id.${num}.volitional.turk.${turk_number} --language=${language}; done
    for num in 1; do python scripts/ConllToErma.py --train=${language}/annotations/train.${num}+turk${turk_number}.conll.train_test --test=${language}/annotations/test.${num}.conll.train_test --model=joint --no-polarity --volitional --note=with_lex_id.no_polarity.${num}.volitional.turk.${turk_number} --language=${language}; done
    for num in 1; do python scripts/ConllToErma.py --train=${language}/annotations/train.${num}+turk${turk_number}.conll.train_test --test=${language}/annotations/test.${num}.conll.train_test --model=pipeline_NE --no-polarity --volitional --note=with_lex_id.no_polarity.${num}.volitional.turk.${turk_number} --language=${language}; done

    for fid in ${language}/qsub/pipeline_NE.*turk.${turk_number}.sh; do qsub -q text.q ${fid}; done
    for fid in ${language}/qsub/joint.*turk.${turk_number}.sh; do qsub -q text.q ${fid}; done
    for fid in ${language}/qsub/linear.*turk.${turk_number}.sh; do qsub -q text.q ${fid}; done


fi

if [ "$command" == train ]; then
    for num in {2..10}; do ./add_features.sh ${language}/10-fold/train.${num} ${language}; ./add_features.sh ${language}/10-fold/test.${num} ${language}; done

    for num in {2..10}; do python scripts/ConllToErma.py --train=${language}/annotations/train.${num}.conll.train_test --test=${language}/annotations/test.${num}.conll.train_test --model=linear --note=with_lex_id.${num} --language=${language}; done

    for num in {2..10}; do python scripts/ConllToErma.py --train=${language}/annotations/train.${num}.conll.train_test --test=${language}/annotations/test.${num}.conll.train_test --model=linear --no-polarity --note=with_lex_id.no_polarity.${num} --language=${language}; done

    for num in {2..10}; do python scripts/ConllToErma.py --train=${language}/annotations/train.${num}.conll.train_test --test=${language}/annotations/test.${num}.conll.train_test --model=joint --volitional --note=with_lex_id.${num}.volitional --language=${language}; done

    for num in {2..10}; do python scripts/ConllToErma.py --train=${language}/annotations/train.${num}.conll.train_test --test=${language}/annotations/test.${num}.conll.train_test --model=pipeline_NE --volitional --note=with_lex_id.${num}.volitional --language=${language}; done

    for num in {2..10}; do python scripts/ConllToErma.py --train=${language}/annotations/train.${num}.conll.train_test --test=${language}/annotations/test.${num}.conll.train_test --model=joint --no-polarity --volitional --note=with_lex_id.no_polarity.${num}.volitional --language=${language}; done

    for num in {2..10}; do python scripts/ConllToErma.py --train=${language}/annotations/train.${num}.conll.train_test --test=${language}/annotations/test.${num}.conll.train_test --model=pipeline_NE --no-polarity --volitional --note=with_lex_id.no_polarity.${num}.volitional --language=${language}; done

    for num in {2..10}; do for fid in ${language}/qsub/pipeline_NE.*${num}*.sh; do qsub -q text.q ${fid}; done; done
    for num in {2..10}; do for fid in ${language}/qsub/joint.*${num}*.sh; do qsub -q text.q ${fid}; done; done
    for num in {2..10}; do for fid in ${language}/qsub/linear.*${num}*.sh; do qsub -q text.q ${fid}; done; done
fi


if [ "$command" == pipeline_sent ]; then
    for num in {2..10}; do python scripts/ConllToErma.py --train=${language}/annotations/train.${num}.conll.train_test --test=${language}/annotations/test.${num}.conll.train_test --volitional --model=pipeline_sent --pipeline-NE-in=${language}/train_test/pipeline_NE.with_lex_id.${num}.volitional.predictions --note=with_lex_id.${num}.volitional  --language=${language}; done
    
    for num in {2..10}; do python scripts/ConllToErma.py --train=${language}/annotations/train.${num}.conll.train_test --test=${language}/annotations/test.${num}.conll.train_test --volitional --model=pipeline_sent --pipeline-NE-in=${language}/train_test/pipeline_NE.with_lex_id.no_polarity.${num}.volitional.predictions --no-polarity --note=with_lex_id.no_polarity.${num}.volitional  --language=${language}; done

    for num in {2..10}; do for fid in ${language}/qsub/pipeline_sent.*${num}*.sh; do qsub -q text.q ${fid}; done; done

fi


if [ "$command" == compute_results ]; then

    echo "Baselines"
    echo "Heuristic baselines"
    for num in {2..10}; do cat ${language}/annotations/test.${num}.conll.train_test | python scripts/strong_baseline-volitional-no_polarity.py --linear > ${language}/train_test/baseheur.linear.${num}.predictions; done
    for num in {2..10}; do cat ${language}/annotations/test.${num}.conll.train_test | python scripts/strong_baseline-volitional-no_polarity.py > ${language}/train_test/baseheur.${num}.predictions; done
    for NOTE in "${conditions[@]}"; do
        for volitional_status in volitional; do
            echo "Joint models, baselines," ${volitional_status}
            for num in {2..10}; do cat ${language}/train_test/joint.${NOTE}.${num}.${volitional_status}.predictions | sed 's/=positive/=neutral/g;s/=negative/=neutral/g;s/=sentiment/=neutral/g' > ${language}/train_test/baseline.joint.${NOTE}.${num}.${volitional_status}.predictions; done; 
            echo "Pipe models, baselines," ${volitional_status}
            for num in {2..10}; do cat ${language}/train_test/pipe_combined.${NOTE}.${num}.${volitional_status}.predictions | sed 's/=positive/=neutral/g;s/=negative/=neutral/g;s/=sentiment/=neutral/g' > ${language}/train_test/baseline.pipe_combined.${NOTE}.${num}.${volitional_status}.predictions; done
        done
        echo "Collapsed models"
        for num in {2..10}; do cat ${language}/train_test/linear.${NOTE}.${num}.predictions | sed -r 's/([BI])positive/\1neutral/g;s/([BI])negative/\1neutral/g;s/([BI])sentiment/\1neutral/g' > ${language}/train_test/baseline.linear.${NOTE}.${num}.predictions; done
    done
fi

if [ "$command" == combine_pipe ]; then
    for NOTE in "${conditions[@]}"; do for num in {2..10}; do python scripts/combine_pipe.py ${language}/train_test/pipeline_NE.${NOTE}.${num}.volitional.predictions ${language}/train_test/pipeline_sent.${NOTE}.${num}.volitional.predictions > ${language}/train_test/pipe_combined.${NOTE}.${num}.volitional.predictions; done; done
fi


if [ "$command" == print_results ]; then
    #echo "Paul results..."
    #for num in 1; do python scripts/calc_accuracy_all.py linear.with_lex_id.paul.${num}.test linear.with_lex_id.paul.${num}.predictions | grep F-score; done | awk '{printf "%s,", $NF; sum+=$NF} END {printf " TOTAL:%s\n", sum/9.0}';
    #for num in 1; do python scripts/calc_accuracy_all.py linear.with_lex_id.paul.${num}.test baseline.linear.with_lex_id.paul.${num}.predictions | grep F-score; done | awk '{printf "%s,", $NF; sum+=$NF} END {printf " TOTAL:%s\n", sum/9.0}';
    for NOTE in "${conditions[@]}"; do
        #echo "MODEL: BaseHeur"
        #for NOTE in "${conditions[@]}"; do
        #    for num in 1; do python scripts/calc_accuracy_all.py joint.${NOTE}.${num}.${volitional_status}.test baseheur.${num}.predictions; done #| grep F-score; done | awk '{printf "%s,", $NF; sum+=$NF} END {printf " TOTAL:%s\n", sum/9.0}'; 
        #done; echo
            #if [ "$3" == joint ]; then
            #for volitional_status in volitional; do
            #echo "Joint-${volitional_status}: scaling up"
            #echo "MODEL: Joint-${volitional_status}-$NOTE";
            #for num in 100 200 300 400 500 600 700 800 986;
            #        do python scripts/calc_accuracy_all.py ${train_test_dir}/pipe_combined.${NOTE}.1.${volitional_status}.turk.${num}.test ${train_test_dir}/pipe_co,bined.${NOTE}.1.${volitional_status}.turk.${num}.predictions; done > tmp #| grep F-score; done | awk '{printf "%s,", $NF; sum+=$NF} END {printf " TOTAL:%s\n", sum/9.0}';
            #echo "Jiang comparison (no lex can't be compared)"
            #for num in 1;
            #    do python scripts/calc_accuracy_all.py joint.${NOTE}.${num}.${volitional_status}.test joint.${NOTE}.${num}.${volitional_status}.predictions --jiang | grep 'Sentiment Accuracy'; done | awk '{printf "%s,", $NF; sum+=$NF} END {printf " TOTAL:%s\n", sum/9.0}'; done; echo
            #exit
            #echo
            #total=`grep "Accuracy:" tmp | wc -l`
            #echo "Comparing " $total " files"
            #cat tmp | grep Accuracy: | awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Accuracy: %s\n", sum/total}';
            #cat tmp | grep Joint: |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Joint: %s\n", sum/total}';
            ##cat tmp | grep "NE precision:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " NE precision: %s\n", sum/total}';
            #cat tmp | grep "NE recall:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " NE recall: %s\n", sum/total}';
            #cat tmp | grep "NE specificity:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " NE specificity: %s\n", sum/total}';
            #cat tmp | grep "Sentiment precision:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Sentiment precision: %s\n", sum/total}';
            #cat tmp | grep "Sentiment recall:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Sentiment recall: %s\n", sum/total}';
            #cat tmp | grep "Sentiment specificity:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Sentiment specificity: %s\n", sum/total}';
            #done; echo

    

            for volitional_status in volitional; do
            echo "Joint-${volitional_status}"
            echo "MODEL: Joint-${volitional_status}-$NOTE"; 
            for num in {2..10}; 
                    do python scripts/calc_accuracy_all.py ${train_test_dir}/joint.${NOTE}.${num}.${volitional_status}.test ${train_test_dir}/joint.${NOTE}.${num}.${volitional_status}.predictions; done > tmp #| grep F-score; done | awk '{printf "%s,", $NF; sum+=$NF} END {printf " TOTAL:%s\n", sum/9.0}';
            #echo "Jiang comparison (no lex can't be compared)"
            #for num in 1;
            #    do python scripts/calc_accuracy_all.py joint.${NOTE}.${num}.${volitional_status}.test joint.${NOTE}.${num}.${volitional_status}.predictions --jiang | grep 'Sentiment Accuracy'; done | awk '{printf "%s,", $NF; sum+=$NF} END {printf " TOTAL:%s\n", sum/9.0}'; done; echo
            echo
            total=`grep "Accuracy:" tmp | wc -l`
            echo "Comparing " $total " files"
            cat tmp | grep Accuracy: | awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Accuracy: %s\n", sum/total}';
            cat tmp | grep Joint: |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Joint: %s\n", sum/total}';
            cat tmp | grep "NE precision:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " NE precision: %s\n", sum/total}';
            cat tmp | grep "NE recall:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " NE recall: %s\n", sum/total}';
            cat tmp | grep "NE specificity:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " NE specificity: %s\n", sum/total}';
            cat tmp | grep "Sentiment precision:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Sentiment precision: %s\n", sum/total}';
            cat tmp | grep "Sentiment recall:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Sentiment recall: %s\n", sum/total}';
            cat tmp | grep "Sentiment specificity:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Sentiment specificity: %s\n", sum/total}';
            done; echo

            for volitional_status in volitional; do
            echo "Joint-BaseNS-${volitional_status}"
            echo "MODEL: Joint-BaseNS-${volitional_status}-$NOTE"
            for num in {2..10}; 
                do python scripts/calc_accuracy_all.py ${train_test_dir}/joint.${NOTE}.${num}.${volitional_status}.test ${train_test_dir}/baseline.joint.${NOTE}.${num}.${volitional_status}.predictions; done > tmp #| grep F-score; done | awk '{printf "%s,", $NF; sum+=$NF} END {printf " TOTAL:%s\n", sum/9.0}'; done; echo
            echo
            total=`grep "Accuracy:" tmp | wc -l`
            echo "Comparing " $total " files"
            cat tmp | grep Accuracy: | awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Accuracy: %s\n", sum/total}';
            cat tmp | grep Joint: |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Joint: %s\n", sum/total}';
            cat tmp | grep "NE precision:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " NE precision: %s\n", sum/total}';
            cat tmp | grep "NE recall:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " NE recall: %s\n", sum/total}';
            cat tmp | grep "NE specificity:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " NE specificity: %s\n", sum/total}';
            cat tmp | grep "Sentiment precision:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Sentiment precision: %s\n", sum/total}';
            cat tmp | grep "Sentiment recall:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Sentiment recall: %s\n", sum/total}';
            cat tmp | grep "Sentiment specificity:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Sentiment specificity: %s\n", sum/total}';
            done; echo
            #fi

            for volitional_status in volitional; do
            echo "Pipe-${volitional_status}"
            echo "MODEL: Pipe-${volitional_status}-$NOTE"
            for num in {2..10}; 
                do python scripts/calc_accuracy_all.py ${train_test_dir}/joint.${NOTE}.${num}.${volitional_status}.test ${train_test_dir}/pipe_combined.${NOTE}.${num}.${volitional_status}.predictions; done > tmp #| grep F-score; done | awk '{printf "%s,", $NF; sum+=$NF} END {printf " TOTAL:%s\n", sum/9.0}';
            total=`grep "Accuracy:" tmp | wc -l`
            echo "Comparing " $total " files"
            #echo "Accuracy: BLAH" 
            cat tmp | grep Accuracy: | awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Accuracy: %s\n", sum/total}';
            #echo "Joint: BLAH" 
            cat tmp | grep Joint: |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Joint: %s\n", sum/total}';
            #echo "NE precision: BLAH" 
            cat tmp | grep "NE precision:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " NE precision: %s\n", sum/total}';
            #echo "NE recall: BLAH" 
            cat tmp | grep "NE recall:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " NE recall: %s\n", sum/total}';
            #echo "NE specificity:  BLAH" 
            cat tmp | grep "NE specificity:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " NE specificity: %s\n", sum/total}';
            #echo "Sentiment precision: BLAH" 
            cat tmp | grep "Sentiment precision:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Sentiment precision: %s\n", sum/total}';
            #echo "Sentiment recall: BLAH" 
            cat tmp | grep "Sentiment recall:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Sentiment recall: %s\n", sum/total}';
            #echo "Sentiment specificity: BLAH" 
            cat tmp | grep "Sentiment specificity:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Sentiment specificity: %s\n", sum/total}';
            done; echo

            for volitional_status in volitional; do
            echo "Pipe-BaseNS-${volitional_status}"
            echo "MODEL: Pipe-BaseNS-${volitional_status}-$NOTE"
            for num in {2..10}; 
                do python scripts/calc_accuracy_all.py ${train_test_dir}/joint.${NOTE}.${num}.${volitional_status}.test ${train_test_dir}/baseline.pipe_combined.${NOTE}.${num}.${volitional_status}.predictions; done > tmp #| grep F-score; done | awk '{printf "%s,", $NF; sum+=$NF} END {printf " TOTAL:%s\n", sum/9.0}'; done; echo
            total=`grep "Accuracy:" tmp | wc -l`
            echo "Comparing " $total " files"
            #echo "Accuracy: BLAH" 
            cat tmp | grep Accuracy: | awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Accuracy: %s\n", sum/total}';
            #echo "Joint: BLAH" 
            cat tmp | grep Joint: |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Joint: %s\n", sum/total}';
            #echo "NE precision: BLAH" 
            cat tmp | grep "NE precision:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " NE precision: %s\n", sum/total}';
            #echo "NE recall: BLAH" 
            cat tmp | grep "NE recall:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " NE recall: %s\n", sum/total}';
            #echo "NE specificity:  BLAH" 
            cat tmp | grep "NE specificity:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " NE specificity: %s\n", sum/total}';
            #echo "Sentiment precision: BLAH" 
            cat tmp | grep "Sentiment precision:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Sentiment precision: %s\n", sum/total}';
            #echo "Sentiment recall: BLAH" 
            cat tmp | grep "Sentiment recall:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Sentiment recall: %s\n", sum/total}';
            #echo "Sentiment specificity: BLAH" 
            cat tmp | grep "Sentiment specificity:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Sentiment specificity: %s\n", sum/total}';
            done; echo
            #fi


        # "Volitional status" does not matter for these guys -- just use the last one.
        volitional_status="volitional"
        echo "Collapsed"
        echo "MODEL: Collapsed-$NOTE"
        for num in {2..10};
            do python scripts/calc_accuracy_all.py ${train_test_dir}/joint.${NOTE}.${num}.${volitional_status}.test ${train_test_dir}/linear.${NOTE}.${num}.predictions --linear; done > tmp 
        #cat tmp | grep MODEL | awk '{printf "%s,", $NF; sum+=$NF} END {printf " MODEL:%s\n", sum/9.0}';
        total=`grep "Accuracy:" tmp | wc -l`
        echo "Comparing " $total " files"
        cat tmp | grep Accuracy: | awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Accuracy: %s\n", sum/total}';
            cat tmp | grep Joint: |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Joint: %s\n", sum/total}';
            cat tmp | grep "NE precision:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " NE precision: %s\n", sum/total}';
            cat tmp | grep "NE recall:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " NE recall: %s\n", sum/total}';
            cat tmp | grep "NE specificity:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " NE specificity: %s\n", sum/total}';
            cat tmp | grep "Sentiment precision:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Sentiment precision: %s\n", sum/total}';
            cat tmp | grep "Sentiment recall:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Sentiment recall: %s\n", sum/total}';
            cat tmp | grep "Sentiment specificity:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Sentiment specificity: %s\n", sum/total}';

        echo

        echo "Collapsed-BaseNS"
        echo "MODEL: Collapsed-BaseNS-$NOTE"
        for num in {2..10}; 
            do python scripts/calc_accuracy_all.py ${train_test_dir}/joint.${NOTE}.${num}.${volitional_status}.test ${train_test_dir}/baseline.linear.${NOTE}.${num}.predictions --linear; done > tmp #| grep F-score; done | awk '{printf "%s,", $NF; sum+=$NF} END {printf " TOTAL:%s\n", sum/9.0}'; done; echo
        echo
            total=`grep "Accuracy:" tmp | wc -l`
            echo "Comparing " $total " files"
            cat tmp | grep Accuracy: | awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Accuracy: %s\n", sum/total}';
            cat tmp | grep Joint: |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Joint: %s\n", sum/total}';
            cat tmp | grep "NE precision:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " NE precision: %s\n", sum/total}';
            cat tmp | grep "NE recall:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " NE recall: %s\n", sum/total}';
            cat tmp | grep "NE specificity:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " NE specificity: %s\n", sum/total}';
            cat tmp | grep "Sentiment precision:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Sentiment precision: %s\n", sum/total}';
            cat tmp | grep "Sentiment recall:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Sentiment recall: %s\n", sum/total}';
            cat tmp | grep "Sentiment specificity:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Sentiment specificity: %s\n", sum/total}';

    done; echo
    #echo "MODEL: Collapsed-BaseHeur" 
    #for NOTE in "${conditions[@]}";
    #    do echo $NOTE;
    #    for num in {2..10};
    #        do python scripts/calc_accuracy_all.py ${train_test_dir}/joint.${NOTE}.${num}.${volitional_status}.test baseheur.linear.${num}.predictions --linear; done > tmp #| grep F-score; done | awk '{printf "%s,", $NF; sum+=$NF} END {printf " TOTAL:%s\n", sum/9.0}'; done; echo
    #        total=`grep "Accuracy:" tmp | wc -l`
    #        cat tmp | grep Accuracy: | awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Accuracy:%s\n", sum/total}';
    #        cat tmp | grep Joint: |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Joint:%s\n", sum/total}';
    #        cat tmp | grep "NE precision:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " NE precision:%s\n", sum/total}';
    #        cat tmp | grep "NE recall:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " NE recall:%s\n", sum/total}';
    #        cat tmp | grep "NE specificity:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " NE specificity:%s\n", sum/total}';
    #        cat tmp | grep "Sentiment precision:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Sentiment precision:%s\n", sum/total}';
    #        cat tmp | grep "Sentiment recall:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Sentiment recall:%s\n", sum/total}';
    #        cat tmp | grep "Sentiment specificity:" |  awk -v total=$total '{printf "%s,", $NF; sum+=$NF} END {printf " Sentiment specificity:%s\n", sum/total}';
    #done; echo
fi

if [ $command == "features" ]; then
    cat ${train_test_dir}/linear.with_lex_id.no_polarity.1-best.ff | python scripts/weight_model.py >${train_test_dir}/linear.with_lex_id.no_polarity.model_rank
    cat ${train_test_dir}/linear.with_lex_id.no_polarity.model_rank | awk '{if ($1=="###") print; else if ($1~/^word_/) print}' > ${train_test_dir}/pipe_NE.with_lex_id.1.word_features
    cat ${train_test_dir}/linear.with_lex_id.no_polarity.model_rank | awk '{if ($1=="###") print; else if ($1~/^word_/) print}' > ${train_test_dir}/linear.with_lex_id.no_polarity.model_rank.word_features
    grep -v "_id_" ${train_test_dir}/linear.with_lex_id.no_polarity.model_rank.word_features > ${train_test_dir}/linear.with_lex_id.no_polarity.model_rank.word_features.no_IDs
fi
