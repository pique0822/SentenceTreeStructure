import argparse
import random

# python3 arithmetic_generator.py --num_examples 1000 --max_integer 100 --fixed_length True --output_file fixed_1e2/testing.txt

parser = argparse.ArgumentParser(description='Generates a set of lines of arithmetic and the solution.')
parser.add_argument('--num_examples',type=int, default=100,
                    help='Integer that defines the number of examples sentences that should be generated using this PCFG.')
parser.add_argument('--max_integer',type=int, default=1e5,
                    help='Positive integer that defines the maximum positive/negative number that a number in the arithmetic is allowed to be.')
parser.add_argument('--fixed_length',type=str, default='False',
                    help='Flag that determines if the numbers will be held to a fixed length.')
parser.add_argument('--output_file',type=str, default='ex_output.txt',
                    help='File that contains the generated lines from this program.')
args = parser.parse_args()
with open(args.output_file,'w+') as out:
    for line in range(args.num_examples):

        max_len = len(str(args.max_integer))

        num_sum = 0

        num_one = random.randint(0,args.max_integer)
        if random.randint(0,1) == 1:
            sign_one = '+'
            num_sum += num_one
        else:
            sign_one = '-'
            num_sum -= num_one

        extra_zeros_one = '0'*(max_len - len(str(num_one)))


        num_two = random.randint(0,args.max_integer)
        if random.randint(0,1) == 1:
            sign_two = '+'
            num_sum += num_two
        else:
            sign_two = '-'
            num_sum -= num_two

        extra_zeros_two = '0'*(max_len - len(str(num_two)))

        if args.fixed_length == 'False':
            out_string = str(num_one)+'+'+str(num_two)+'=,'+str(num_sum)+'\n'
        else:
            out_string = sign_one+extra_zeros_one+ str(num_one)+'+'+sign_two+extra_zeros_two+str(num_two)+'=,'+str(num_sum)+'\n'
        out.write(out_string)
