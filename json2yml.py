import json
import sys
from pickle import TRUE

from jsonpath_ng import jsonpath, parse

if len(sys.argv) != 2:
    print("Usage: python3 " + sys.argv[0] + " file.json > file.yml")
    exit(1)

fname = sys.argv[1]
with open(fname, "r") as json_file:
    json_data = json.load(json_file)
utterance = parse("turns[*].utterance")
num = 0
u = []
t = []
output_string = "conversations:"
category = "categories:\n- {}".format("ticktock")
print(category)
print(output_string)
for match in utterance.find(json_data):
    if (num % 2) == 0:
        u.append(match.value)
    else:
        t.append(match.value)
    num += 1

for i in range(len(u)):
    s = u[i]
    s1 = t[i]
    output_string = "- - {}\n  - {}\n".format(s, s1)
    print(output_string, end="")
print()
