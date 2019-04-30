import sys
import re

fid = sys.stdin.readlines()

x = 0
for line in fid[2:]:
    if x%4 == 0:
        split_line = line.split()
        if len(split_line) > 1:
            #sys.stderr.write("Changing " + line.strip() + " to null -- is that ok?\n")
            line = "null\n"
        print line,
    x += 1
