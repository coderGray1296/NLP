import sys
import re

NE = open(sys.argv[1], "r").readlines()
sent = open(sys.argv[2], "r").readlines()

def read_in(fid, example_hash={}):
    example = 0
    for line in fid:
        line = line.strip()
        if line.startswith("example"):
            example += 1
        elif re.search("o=", line):
            split_line = line.split("=")
            var = split_line[0]
            val = split_line[1]
            try:
                example_hash[example][var] = val
            except KeyError:
                example_hash[example] = {var:val}
    return example_hash

def write_out(example_hash):
    for example in sorted(example_hash):
        print "//example " + str(example)
        print "example:"
        for RV in example_hash[example]:
            print RV + "=" + example_hash[example][RV]

out_hash = read_in(NE)
out_hash = read_in(sent, out_hash)
write_out(out_hash)
