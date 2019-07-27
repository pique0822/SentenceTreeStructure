from nltk import PCFG
from nltk.parse.generate import generate
from heapq import nlargest
import random

import argparse

parser = argparse.ArgumentParser(description='Generates a set of sentences as defined by some set of PCFG rules.')
parser.add_argument('--rule_file', type=str, default='language_rules/rules.cfg',
                    help='File path to rule file that describes how to generate language corpus.')
parser.add_argument('--num_examples',type=int, default=100,
                    help='Integer that defines the number of examples sentences that should be generated using this PCFG.')
parser.add_argument('--output_file',type=str, default='ex_output.txt',
                    help='File that contains the generated sentences from this program.')
args = parser.parse_args()

# can specify depth if needed
def generate_sample(grammar):
    rules = grammar._lhs_index
    start = grammar._start

    import pdb; pdb.set_trace()

grammar_text = ""

with open(args.rule_file) as file:
    for line in file:
        grammar_text += line

grammar = PCFG.fromstring(grammar_text)

generate_sample(grammar)

sentences = generate(grammar, n=args.num_examples)

with open(args.output_file, 'w+') as file:
    for s in sentences:
        file.write(' '.join(s) + '\n')
