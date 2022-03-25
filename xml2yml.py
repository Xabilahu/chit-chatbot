#!/usr/bin/python3

# Convert a brown tei document to a text file, where each sentence is a different line.
# Each token in the sentence must have the format "word/POS".
#
# For instance,
#
# The/AT Fulton/NP County/NN Grand/JJ Jury/NN said/VBD Friday/NR an/AT investigation/NN of/IN Atlanta's/NPg recent/JJ primary/NN election/NN produced/VBD ``/pct no/AT evidence/NN ''/pct that/CS any/DTI irregularities/NNS took/VBD place/NN ./pct
# The/AT jury/NN further/RBR said/VBD in/IN term-end/NN presentments/NNS that/CS the/AT City/NN Executive/JJ Committee/NN ,/pct which/WDT had/HVD over-all/JJ charge/NN of/IN the/AT election/NN ,/pct ``/pct deserves/VBZ the/AT praise/NN and/CC thanks/NNS of/IN the/AT City/NN of/IN Atlanta/NP ''/pct for/IN the/AT manner/NN in/IN which/WDT the/AT election/NN was/BEDZ conducted/VBN ./pct
#
#   * you can use the script do_brown_txt.sh that will execute this script
#     with every brown xml document and convert them to text

import sys
from asyncio import subprocess
from re import sub

from lxml import etree

if len(sys.argv) != 2:
    print("Usage: python3 " + sys.argv[0] + " file.xml > file.yml")
    exit(1)

fname = sys.argv[1]
parser = etree.XMLParser(remove_blank_text=True)  # discard whitespace nodes
tree = etree.parse(fname, parser)
output_setence = ""
for elem in tree.xpath("//document/s"):
    subelem = elem.get("id")
    subelemc = elem.text
    print("s", subelemc)

    for subelem in elem:

        postag = subelem.get("id")
        print(postag)
        token = subelem.text
        print("t", token)

        # output_string = "{}\\{}".format(token.strip(), postag)
        # print(output_string,end=" ")
print()
