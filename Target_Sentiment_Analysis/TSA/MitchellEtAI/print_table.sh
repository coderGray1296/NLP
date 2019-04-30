#!/bin/bash


langa=$1
tmp3=$2
#./run_EMNLP.sh $langa print_results > tmp3
echo -n " {\\bf Model} & "; cat $tmp3 | grep MODEL | awk '{printf "%.4f ", $NF}' | awk '{for(i=1;i<=NF/2.0;i++) if (i%2!=0||i==NF/2.0)printf "%s & ", $i} END {print "\\\\"}'
echo -n " {\\bf Acc-all} & "; cat $tmp3  | grep Accuracy: | awk '{printf "%.4f ", $NF}' | awk '{for(i=1;i<=NF/2.0;i++) if (i%2!=0||i==NF/2.0) printf "%.4f & ", $i} END {print "\\\\"}'
echo -n " {\\bf Acc-Bsent} & "; cat $tmp3  | grep Joint: | awk '{printf "%.4f ", $NF}' | awk '{for(i=1;i<=NF/2.0;i++) if (i%2!=0||i==NF/2.0)printf "%.4f & ", $i} END {print "\\\\\\hline"}'
echo -n " {\\bf NE prec} & "; cat $tmp3 | grep "NE precision:" | awk '{printf "%.4f ", $NF}' | awk '{for(i=1;i<=NF/2.0;i++) if (i%2!=0||i==NF/2.0)printf "%.4f & ", $i} END {print "\\\\"}'
echo -n " {\\bf NE rec} & "; cat $tmp3 | grep "NE recall:" | awk '{printf "%.4f ", $NF}' | awk '{for(i=1;i<=NF/2.0;i++) if (i%2!=0||i==NF/2.0)printf "%.4f & ", $i} END {print "\\\\"}'
echo -n " {\\bf NE spec} & "; cat $tmp3 | grep "NE specificity:" | awk '{printf "%.4f ", $NF}' | awk '{for(i=1;i<=NF/2.0;i++) if (i%2!=0||i==NF/2.0)printf "%.4f & ", $i} END {print "\\\\\\hline"}'
echo -n " {\\bf Sent prec} & "; cat $tmp3 | grep "Sentiment precision:" | awk '{printf "%.4f ", $NF}' | awk '{for(i=1;i<=NF/2.0;i++) if (i%2!=0||i==NF/2.0)printf "%.4f & ", $i} END {print "\\\\"}'
echo -n " {\\bf Sent rec} & "; cat $tmp3 | grep "Sentiment recall:" | awk '{printf "%.4f ", $NF}' | awk '{for(i=1;i<=NF/2.0;i++) if (i%2!=0||i==NF/2.0)printf "%.4f & ", $i} END {print "\\\\"}'
echo -n " {\\bf Sent spec} & ";cat $tmp3 | grep "Sentiment specificity:" | awk '{printf "%.4f ", $NF}' | awk '{for(i=1;i<=NF/2.0;i++) if (i%2!=0||i==NF/2.0)printf "%.4f & ", $i} END {print "\\\\"}'

echo; echo

echo -n " {\\bf Model} & "; cat $tmp3 | grep MODEL | awk '{printf "%.4f ", $NF}' | awk '{for(i=NF/2.0+1;i<=NF;i++) if (i%2!=0||i==NF)printf "%s & ", $i} END {print "\\\\"}'
echo -n " {\\bf Acc-all} & "; cat $tmp3 | grep Accuracy: | awk '{printf "%.4f ", $NF}' | awk '{for(i=NF/2.0+1;i<=NF;i++) if (i%2!=0||i==NF)printf "%.4f & ", $i} END {print "\\\\"}'
echo -n " {\\bf Acc-Bsent} & "; cat $tmp3 | grep Joint: | awk '{printf "%.4f ", $NF}' | awk '{for(i=NF/2.0+1;i<=NF;i++) if (i%2!=0||i==NF)printf "%.4f & ", $i} END {print "\\\\\\hline"}'
echo -n " {\\bf NE prec} & "; cat $tmp3 | grep "NE precision:" | awk '{printf "%.4f ", $NF}' | awk '{for(i=NF/2.0+1;i<=NF;i++) if (i%2!=0||i==NF)printf "%.4f & ", $i} END {print "\\\\"}'
echo -n " {\\bf NE rec} & "; cat $tmp3 | grep "NE recall:" | awk '{printf "%.4f ", $NF}' | awk '{for(i=NF/2.0+1;i<=NF;i++) if (i%2!=0||i==NF)printf "%.4f & ", $i} END {print "\\\\"}'
echo -n " {\\bf NE spec} & "; cat $tmp3 | grep "NE specificity:" | awk '{printf "%.4f ", $NF}' | awk '{for(i=NF/2.0+1;i<=NF;i++) if (i%2!=0||i==NF)printf "%.4f & ", $i} END {print "\\\\\\hline"}'
echo -n " {\\bf Sent prec} & "; cat $tmp3 | grep "Sentiment precision:" | awk '{printf "%.4f ", $NF}' | awk '{for(i=NF/2.0+1;i<=NF;i++) if (i%2!=0||i==NF)printf "%.4f & ", $i} END {print "\\\\"}'
echo -n " {\\bf Sent rec} & "; cat $tmp3 | grep "Sentiment recall:" | awk '{printf "%.4f ", $NF}' | awk '{for(i=NF/2.0+1;i<=NF;i++) if (i%2!=0||i==NF)printf "%.4f & ", $i} END {print "\\\\"}'
echo -n " {\\bf Sent spec} & "; cat $tmp3 | grep "Sentiment specificity:" | awk '{printf "%.4f ", $NF}' | awk '{for(i=NF/2.0+1;i<=NF;i++) if (i%2!=0||i==NF)printf "%.4f & ", $i} END {print "\\\\"}'

