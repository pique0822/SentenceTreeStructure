# Defining .rul Rules

A .rul file should be defined as follows:
vocab::{V}
rule1
rule2
...

Such that V is the comma separated set of characters that can be used in the language and each line contains one rule of the artificial language to be generated. Do note that rules are not expected to be chained as in the two rules A$B and B$C are a set of invalid rules since these two would cause a chain.

Example:
vocab::{A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z}
A$B

## Symbols that can be used
1. A$B - this symbol implies a follow requirement. When we see the sequence 'A' we require that it be followed by 'B'
2.
