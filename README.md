# Huffman Encoding

## Objectives

* Build trees
* Traverse trees
* Write code to a given interface
* Identify own helper methods and writing them
* Choose, design and implement appropriate data structures where needed

# Overview
In this project we’re concerned with communicating between two computers. We can only use 0’s and 1’s to communicate between two computers– which ultimately get transmitted as pulses of electricity across a wire. Thus, we have to figure out how to take everything that we want to communicate (as humans), convert it into a series of 0’s and 1’s, which are converted into electronic pulses, which can be sent to another receiving device, which converts those electric pulses into 0’s and 1’s, and then reconstruct those 0’s and 1’s into letters, and words.

To understand Huffman encoding and its significance, we need to understand how text is generally represented by computers.

## Basic Text Encoding
Digital text is encoded with ASCII. Each character is represented by an integer in a range (0-127). That character is represented in 8 bits (0’s and 1’s) by converting the decimal number to a binary number. For reference, http://www.asciitable.com/ is a resource to see the mapping between characters and ASCII values.

For example, the word “Bike”:

#### Table 1: ASCII Values for ‘Bike’
| Character | ASCII Val | (Decimal) Binary |
| ----------- | ----------- | ----------- |
| B | 66 | 01000010 |
| i | 105 | 01101001 |
| k | 107 | 01101011 |
| e | 101 | 01100101 |

That means the sequences of bits that represent the word ‘Bike’ is:

#### Table 2: Bit sequence for ‘Bike’
| 01000010 | 01101001 | 01101011 | 01100101 |
| ----------- | ----------- | ----------- | ----------- |

Computers represent the word “Bike” by a bunch of bits: `01000010|01101001|01101011|01100101`. (The pipe is just showing the letter boundaries).

The length of this bit stream is always 8∗ [the number of letters]: for “Bike”, we always need 32 bits to communicate “Bike”. Can we improve on this? How much?

##  Improving on basic text encoding: Reduce the number of bits

The goal is to reduce the number of bits we use to represent the word ‘Bike’. Since we only have 4 letters in the word ‘Bike’, we could easily do something like:

#### Table 3: Minimal Encoding for “Bike”
| Character | (Decimal) Binary |
| ----------- | ----------- |
| B | 00 |
| i | 01 |
| k | 10 |
| e | 11 |

Using this mapping the sequence of bits we transmit is “00-01-10-11”– only 8 bits instead of the previous 32.

This example is almost trivial: it’s only 4 characters, and none of them are duplicated. Let’s look at a possible encoding for “Bicycle”, which has multiple ‘c’s in it (Table 4).


#### Table 4: Variable Length Encoding for “Bicycle”
| Character | (Decimal) Binary |
| ----------- | ----------- |
| B | 010 |
| i | 011 |
| c | 10 |
| y | 110 |
| l | 111 |
| e | 00 |


The sequence of bits with this mapping is “010-011-10-110-10-111-00”– again, only 18 bits, instead of the original 48 needed in the traditional approach. Note that in this mapping, the letter ‘c’ appears twice in the original word, and has the shortest number of bits to represent it. By mapping ‘c’ to a shorter sequence of bits (in this case, 2 instead of 3), we save 2 bits in our output.

However, the receiver also needs to know the mapping. The person on the other end has no idea how many bits is representing a single character (so they don’t know where the character boundaries are), and also which bit sequence maps to which character. Without this, the receiver doesn’t know how to decode ‘010-011-10-...’ to ‘Bike’. In the standard approach, the receiver knows that each sequence of 8 bits maps to a character, and has an agreed-upon map to translate the bits to a character (that ASCII table).

## Huffman Encoding

Huffman encoding takes a similar approach to what is described above:

1. Figure out which characters are in the original string/word/document (and their frequencies,
but we’ll get to that later)
2. Determine a sequence of bits to represent each character in the input
3. Encode the string/word/document using this mapping
4. Transmit the sequence of bits to the receiver
5. Transmit/share the mapping with the receiver
6. Decode the sequence of bits into a message using the mapping

The pieces we’re concerned with as a programmer (that is, for this assignment):

* Calculating characters & frequencies
* Creating the “mapping” — the Huffman tree
* Encoding a message with the mapping
* Decoding a message with the mapping

###  Calculating characters & frequencies

This part of the process should be straightforward. We’re just looking to create a frequency table from the given input.

#### Table 5: Frequency Table for “Bike”
| Character | Frequency |
| ----------- | ----------- |
| B | 1 |
| i | 1 |
| k | 1 |
| e | 1 |

For the input ‘Bike’, the table is trivial (see Table 5). 

For this assignment, the frequency table struct is as follows:

```python
def __init__(self, input_str):
  self.char_count = [0] * 256
```

The array `char_count` represents the counts of each character. The index corresponds to the ASCII value. 
For example, the ASCII value of ‘a’ is 97. `char_count[97]` represents the count of a’s in the input string.

The function that populates the frequency table is:

```python
def populate(self, input_str):
```

###  Creating the mapping: The Huffman Tree

With Huffman encoding, we create a tree to encode and decode a characters. The bit sequence to represent a character is simply the path one takes through the tree.

Further, we can use this tree to go from `00011011` to the string “Bike” by tracing each bit through the tree. 
The first bit is 0, so we start at the root and take the left branch, labeled 0. There is no character at that point, 
so we take the next bit, and go left again. There is a character, so we now have the first character, ‘B’. We go back 
to the root, and repeat the process starting with the next character, which is also a ‘0’.

The tree is created thusly:

* Create a node for each character (including the count), and put it in an appropriate data
structure
* While there is more than 1 node in the list:
  * Select two nodes with the lowest counts
  * Create a new node that is the parent node of those two nodes
  * Set the count of that new node to be the sum of the counts of the two children nodes
  * Take the two children nodes out of the list
  * Put the new parent node in the list
  * The last node in the list is the root of the tree.
  
The data structure to use for your tree is defined as:

```python
class HTree:
    def __init__(self, c=None, freq=0, p0=None, p1=None):
        self.c = c
        self.freq = freq
        self.p0 = p0
        self.p1 = p1
```

* `c` is the character represented by this node. If this node is an internal node (at least one of `p0` or `p1` is not `None`), `c` is 0.
* `freq` is the frequency or count of characters reachable by this node. If this node is a leaf (thereby representing a single character) it’s merely the frequency of that character in the input document. If this node is an internal node, it’s the sum of the frequencies of the children of this node.
* `p0`: The left or `0` child of this node (think "path0").
* `p1`: The right or `1` child of this node (think "path1").

The function that will create the tree is as follows:

```python
create_encoding_tree(freq_table)
```
`create_encoding_tree` should build the tree with the given frequencies.

###  Encode a message

Once we have a Huffman tree, the encoding is straightforward. Each character maps to a sequence of bits that represent 
a traversal through the tree. 

The most efficient way to do the encoding is to generate a table such as Table 4. You can do this by traversing the 
tree one time. This approach is known as creating a “lookup table”. Rather than performing a costly calculation every 
time we do something, we do the costly calculation one time, cache or store the results, and then use the table to 
look up the answer.

Thus, the process is:

1. Create the lookup table. What data structure might you want to use? You will want to use a traversal. Which traversal
will be best for this? You will need to generate all paths in the tree.
2. For each character in the string:
   (a) Look up the sequence of bits in the lookup table
   (b) Append the bits to the output sequence
4. Return the final bit sequence


###  Decode a message

To decode, we start with a sequence of bits and a Huffman tree.

Go through each bit in the sequence, and let that be your traversal through the tree. If you hit a character, emit 
that character and start a new traversal with the next bit in the sequence.

1. For each bit in the sequence:
   * Start at the root of the tree.
   * If the bit == 0, go to the left; otherwise go to the right.
2. If the node has a character, emit the character. Otherwise, keep traversing.
3. Start at the root of the tree again, with the next bit in the sequence.

# Making Binary Strings with `BitSeq`

We've been talking about bits and chars and strings and such: bits make up a char, and chars make up a string.
In the bigger picture of Huffman Encoding, we use the encoding to "pack bits". However, for this assignment, 
we're going to keep it a little simpler. 

`BitSeq` is a class that will help us keep track of our 1s and 0s. In actuality, it's just a list of 
characters, but all those characters are 1s and 0s.

```python
class BitSeq:
    MAX_BITS_PER_INT = 16

    def __init__(self, max_bits_per_int=16):
        self.bits = []  # List of ints-- keep them 16-bit unsigned
        self.num_bits_in_seq = 0
        self.MAX_BITS_PER_INT = max_bits_per_int
```

* `BitSeqStr.bits` is a list of chars ('1's and '0's) that we are using to hold bits.
* `BitSeq.num_bits_in_seq` keeps track of the number of bits we have overall. 
* `BitSeq.pack_bits(str)` takes a string of "1"s and "0"s add packs them into the `bits` list. 
* `BitSeq.get_bit(which_bit)` returns the specified bit as a "1" or "0".
* `BitSeq.get_bits_as_string()` returns the bits put together into a single string of 1s and 0s. 

You'll see in the starter code that there are some things that we'll do to try to make it easier to 
work with the sequence; specifically, when we put the "bits" together into a string, insert spaces. 

With luck, implementing `BitSeq` will be pretty straightforward, and you'll be wondering why 
we are bothering with some of this :) 

If you are looking for more of a challenge and are interested in reducing our memory footprint, 
reach out to Adrienne for an alternate approach! 

# Your Task

Populate `huffman.py` by running the tests as you go.  
You may find it helpful to define helper functions.

`huffman_test.py` has some starter tests. They should pass as you start filling out your 
code. Some of the tests do depend on specific implementation details; they **should** 
run correctly when you write your code to the provided specification, but 
if you do see a test that depends on a small implementation detail that is different 
from what you used, feel free to change it. 

##  Hopefully Helpful Hints

* Use the defined constants
* Write your own tests! 
   * If the provided tests aren't helping you figure out what is not working right, write another one! 
* If you are struggling with breaking your project down into smaller pieces, talk to an instructor/TA. Spend some time thinking about it, but feel free to ask for guidance on this part.
* Write “print” helper functions. Again, we can’t write them all for you, because your implementation might vary. We’ve provided print functions where we could. Use those as guidance,
and ask for help.
* Specifically for creating the `HTree`, consider other data structures we’ve seen, and feel free to create/implement one to help with the process of creating the `HTree`.
* Design and implement your own structs that help you solve the problem.
* Start early. Take small steps.


It might help to approach the assignment in this order:

1. Implement `BitSeqStr`:
   * Print a `BitSeqStr`
   * Add bits to a `BitSeqStr`
   * Get a single bit from a `BitSeqStr`
2. Implement `FreqTable`:
   * print a `FreqTable`
   * populate the `FreqTable`
3. Implement `HTree`
4. Implement `LookupTable`
5. Implement `Encoder`
6. Implement `Decoder`

### Resources:

* https://realpython.com/python-bitwise-operators: A lot of gems on this page, but also a lot of detail. 
* A GREAT short explainer with context of why this matters: https://www.youtube.com/watch?v=JsTptu56GM8
* A shorty that walks through the process of creating and using a small Huffman tree by hand: https://www.youtube.com/watch?v=iEm1NRyEe5c
