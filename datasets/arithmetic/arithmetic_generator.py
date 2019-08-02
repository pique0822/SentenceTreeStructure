import argparse
import random

# python3 arithmetic_generator.py --num_examples 1000 --max_integer 1000 --fixed_length True --numbers_per_sum 4 --output_file fixed_L4_1e3/testing.txt

parser = argparse.ArgumentParser(description='Generates a set of lines of arithmetic and the solution.')
parser.add_argument('--num_examples',type=int, default=100,
                    help='Integer that defines the number of examples sentences that should be generated using this PCFG.')
parser.add_argument('--max_integer',type=int, default=1e5,
                    help='Positive integer that defines the maximum positive/negative number that a number in the arithmetic is allowed to be.')
parser.add_argument('--fixed_length',type=str, default='False',
                    help='Flag that determines if the numbers will be held to a fixed length.')
parser.add_argument('--output_file',type=str, default='ex_output.txt',
                    help='File that contains the generated lines from this program.')
parser.add_argument('--numbers_per_sum', type=int,
                    default=2, help='The quantity of numbers to add in a line.')
args = parser.parse_args()
with open(args.output_file,'w+') as out:
    for line in range(args.num_examples):

        max_len = len(str(args.max_integer))

        num_sum = 0
        out_string = ""

        for num_index in range(args.numbers_per_sum):
            this_num = random.randint(0,args.max_integer)
            if random.randint(0,1) == 1:
                this_sign = '+'
                num_sum += this_num
            else:
                this_sign = '-'
                num_sum -= this_num

            extra_zeros = '0'*(max_len - len(str(this_num)))

            if args.fixed_length == 'True':
                out_string += this_sign+extra_zeros+ str(this_num)+'+'
            else:
                out_string += this_sign+ str(this_num)+'+'

        out_string = out_string[:len(out_string)-1]+'=,'+str(num_sum)+'\n'
        out.write(out_string)
