import sys
import re

fid = sys.stdin.readlines()
go = False
feature_hash = {}
feature_list = {}
f = 0
for line in fid:
    if line.startswith("features:"):
        go = True
    if go:
        split_line = line.split("=")
        if len(split_line) == 1:
            continue
        else:
            f += 1
            weight = float(split_line[-1])
            feature = split_line[0]
            NE = split_line[1]
            try:
                feature_list[NE][f] = (weight, feature)
            except KeyError:
                feature_list[NE] = {f:(weight, feature)}

for NE in feature_list:
    print "### NE: ", NE
    vals = feature_list[NE].values()
    vals.sort()
    for (weight, feature) in vals:
        print feature + "\t\t" + str(weight)
