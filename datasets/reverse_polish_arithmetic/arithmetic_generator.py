import argparse
import random
import numpy as np

# python3 arithmetic_generator.py --num_examples 9000 --max_integer 100 --fixed_length True --output_file fixed_L2_1e2/training.txt

parser = argparse.ArgumentParser(description='Generates a set of lines of arithmetic and the solution.')
parser.add_argument('--num_examples',type=int, default=100,
                    help='Integer that defines the number of examples sentences that should be generated using this PCFG.')
parser.add_argument('--max_integer',type=int, default=1e2,
                    help='Positive integer that defines the maximum positive/negative number that a number in the arithmetic is allowed to be.')
parser.add_argument('--fixed_length',type=str, default='False',
                    help='Flag that determines if the numbers will be held to a fixed length.')
parser.add_argument('--output_file',type=str, default='ex_output.txt',
                    help='File that contains the generated lines from this program.')
args = parser.parse_args()
with open(args.output_file,'w+') as out:
    for line in range(args.num_examples):
        select_operator = random.randint(0,4)




        max_len = len(str(args.max_integer))
        out_string = ""



        num_one = 0
        this_num = random.randint(0,args.max_integer)
        if random.randint(0,1) == 1:
            this_sign = '+'
            num_one = this_num
        else:
            this_sign = '-'
            num_one = -this_num

        extra_zeros = '0'*(max_len - len(str(this_num)))

        if args.fixed_length == 'True':
            out_string += this_sign+extra_zeros+ str(this_num)+' '
        else:
            out_string += this_sign+ str(this_num)+' '


        num_two = 0
        this_num = random.randint(0,args.max_integer)
        if random.randint(0,1) == 1:
            this_sign = '+'
            num_two = this_num
        else:
            this_sign = '-'
            num_two = -this_num

        extra_zeros = '0'*(max_len - len(str(this_num)))

        if args.fixed_length == 'True':
            out_string += this_sign+extra_zeros+ str(this_num)+' '
        else:
            out_string += this_sign+ str(this_num)+' '

        # solve arithemtic
        result = 0

        if num_two == 0 and select_operator == 2:
            select_operator = np.random.choice([0,1,3])

        if select_operator == 0:
            operator = '+'
            result = num_one + num_two
        elif select_operator == 1:
            operator = '-'
            result = num_one - num_two
        elif select_operator == 2:
            operator = '/'
            result = num_one / num_two
        else:
            operator = '*'
            result = num_one * num_two


        out_string += operator + ','+str(result)+'\n'
        out.write(out_string)
