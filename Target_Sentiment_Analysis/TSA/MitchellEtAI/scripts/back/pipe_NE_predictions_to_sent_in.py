import sys
import re
from collections import defaultdict

class Pipeline:
    def __init__(self, orig_test, NE_predictions, new_test):
        self.orig_hash = defaultdict(lambda: defaultdict(lambda: defaultdict(defaultdict)))
        self.NE_hash = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
        self.read_orig_test(orig_test)
        self.read_NE_predictions(NE_predictions)
        self.write_out(new_test)


    def norm_answers(self, answers):
        for example in answers:
            for RV in answers[example]:
                val = answers[example][RV]
                if val.startswith("I"):
                    desired_val = re.sub("^I", "B", val)
                    one_before_last = RV
                    prev_RV = "W" + str(int(RV.strip("W")) - 1)
                    if prev_RV in answers[example]:
                        prev_val = answers[example][prev_RV]
                        while prev_val == val:
                            one_before_last = prev_RV
                            one_before_last_val = prev_val
                            prev_RV = "W" + str(int(prev_RV.strip("W")) - 1)
                            if prev_RV in answers[example]:
                                prev_val = answers[example][prev_RV]
                            else:
                                prev_val = None
                        if prev_val != desired_val:
                            #sys.stderr.write(RV + " " + val + " " + prev_RV + " " + prev_val + "\n")
                            #new_val = re.sub("I_", "B_", val)
                            sys.stderr.write("Example " + str(example) + ": Changing " + one_before_last + " to " + desired_val + "\n")
                            answers[example][one_before_last] = desired_val
                    else:
                        answers[example][RV] = desired_val
        return answers



    def read_orig_test(self, orig_test):
        example = -1
        for line in orig_test:
            strip_line = line.strip()
            if strip_line == "":
                continue
            if line.startswith("example"):
                example += 1
                continue
            if line.startswith("//Tweet"):
                continue
            if line.startswith("features:"):
                n = 0
                continue
            strip_line = strip_line.strip(";")
            split_line = strip_line.split()
            if len(split_line) > 1:
                assignment = split_line[1]
                split_ass = assignment.split("=")
                RV = split_ass[0]
                val = split_ass[1]
                if len(split_line) > 2:
                    if not split_line[-1].endswith("in"):
                        sys.stderr.write("Error:" + strip_line + "\n")
                        sys.exit()
                    self.orig_hash["in"][example][RV] = val
                else:
                    self.orig_hash["out"][example][RV] = val
            else:
                self.orig_hash["features"][example][n][strip_line + ";"] = {}
                n += 1

    def read_NE_predictions(self, NE_predictions):
        example = -1
        for line in NE_predictions:
            strip_line = line.strip()
            if strip_line == "":
                continue
            if line.startswith("//"):
                continue
            if line.startswith("example"):
                example += 1
                continue
            split_ass = strip_line.split("=")
            RV = split_ass[0][:-1]
            val = split_ass[1]
            val = val.split()
            if len(val) > 1:
                val = val[0]
            self.NE_hash["out"][example][RV] = val
        self.NE_hash["out"] = self.norm_answers(self.NE_hash["out"])

    def write_out(self, new_test):
        print "Writing out..."
        for example in sorted(self.orig_hash["out"]):
            new_test.write("//Tweet " + str(example + 1)  + "\n")
            new_test.write("example:"  + "\n")
            for RV in self.orig_hash["in"][example]:
                # Last output becomes current input.
                new_test.write("NE " + RV + "=" + self.NE_hash["out"][example][RV] + " in;"  + "\n")
            for RV in self.orig_hash["out"][example]:
                new_test.write("SENT " + RV + "=" + self.orig_hash["out"][example][RV] + ";"  + "\n")
            new_test.write("features:"  + "\n")
            for n in sorted(self.orig_hash["features"][example]):
                # Just one.
                for feature in self.orig_hash["features"][example][n]:
                    new_test.write(feature + "\n")
            new_test.write("\n")
