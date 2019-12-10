# -*- coding: utf-8 -*-
"""Created on Tue Nov 26 14:21:21 2019@author: Alexm"""
"""
# Argparse is used for command line interface



 # Now, via the command line, we can do something like:
    
    python argparse_example.py --x=5 --y=3 --operation=mul
    
 # You should get 15.0 returned via the command line in this case. Another thing we can do is:
    
    python argparse_example.py -h
    
 # Output:
 #  
 #  usage: argparse_example.py [-h] [--x X] [--y Y] [--operation OPERATION]
 #  
 #  optional arguments:
 #    -h, --help            show this help message and exit
 #    --x X                 What is the first number?
 #    --y Y                 What is the second number?
 #    --operation OPERATION
 #                          What operation? Can choose add, sub, mul, or div


"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--x', type=float, default=1.0,help='What is the first number?')
    parser.add_argument('--y', type=float, default=1.0,help='What is the second number?')
    parser.add_argument('--operation', type=str, default='add',
                        help='What operation? Can choose add, sub, mul, or div')
    args = parser.parse_args()
    sys.stdout.write(str(calc(args)))
    
def calc(args):
    if args.operation == 'add':
        return args.x + args.y
    elif args.operation == 'sub':
        return args.x - args.y
    elif args.operation == 'mul':
        return args.x * args.y
    elif args.operation == 'div':
        return args.x / args.y

if __name__ == '__main__':
    main()