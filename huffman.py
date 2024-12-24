

class BitSeq:
    """A BitSeq is a sequence of bits

    Represented by a list of numbers that hold the packed bits,
    and a bunch of helper methods to help us build up a bit
    sequence and print/manipulate/observe the sequence of bits.
    """
    MAX_BITS_PER_INT = 16

    def __init__(self, max_bits_per_int=16):
        self.bits = []  # List of ints-- keep them 16-bit unsigned
        self.num_bits_in_seq = 0
        self.MAX_BITS_PER_INT = max_bits_per_int

    ## Only returns the first num_bits_in_seq characters
    ## e.g. get_bits_as_string() returns "1111"
    def get_bits_as_string(self):
        """Returns a string that represents the bits stored in this BitSeq.

        A space should be included after every MAX_BITS_PER_INT ints.
        If MAX_BIT_PER_INT = 4, a BitSeq would be returned as 1111 0000 0101 1001
        (with a break every 4 bits)
        """
        bits_as_string = ""
        for i in range(self.num_bits_in_seq):
            bits_as_string += str(self.get_bit(i))
            if (i + 1) % self.MAX_BITS_PER_INT == 0:
                bits_as_string += " "
        return bits_as_string.strip()

    ## e.g. pack_bits("1111") will put the bits 1111 in the first available spot
    def pack_bits(self, new_bits_as_chars: str):
        """Given a string of 1s and 0s, packs relevant bits into this BitSeq"""
        ## For each bit/char in the input string:
        ##  determine if we need a new int in self.bits to hold more bits
        ##  Add a new bit to the last int
        for bit in new_bits_as_chars:
            if bit == ' ':
                continue  # Skip spaces in the input string
            if self.num_bits_in_seq % self.MAX_BITS_PER_INT == 0:
                self.bits.append(0)
            if bit == '1':
                self.bits[-1] |= (1 << (self.num_bits_in_seq % self.MAX_BITS_PER_INT))
            self.num_bits_in_seq += 1

    def get_bit(self, which_bit: int) -> int:
        """Get the bit at position which_bit; 0-based index"""
        ## 0-based indexing
        ## If which_bit >= num_bits_in_seq throw an IndexError
        if which_bit >= self.num_bits_in_seq:
            raise IndexError
        int_index = which_bit // self.MAX_BITS_PER_INT
        bit_index = which_bit % self.MAX_BITS_PER_INT
        bit_value = 1 if (self.bits[int_index] & (1 << bit_index)) else 0
        return bit_value


class FreqTable:
    """A table that holds the frequency count of each character. """
    def __init__(self, input_str: str = ""):
        self.char_count = [0] * 256
        self.order = []
        self.populate(input_str)

    def clear(self):
        """Resets the frequency counts such that all frequencies are 0"""
        self.char_count = [0] * 256
        self.order = []

    def populate(self, input_str):
        """Given an input_str, update the frequency of each character in the table according to the string. """
        for c in input_str:
            if self.char_count[ord(c)] == 0:
                self.order.append(c)  # Add character to order list if it's the first occurrence
            self.char_count[ord(c)] += 1

    def get_char_count(self, char):
        """Returns the current frequency count for the given char"""
        return self.char_count[ord(char)]

    def print_freq_table(self):
        """Print the frequency table in an easy to view format"""
        for c in self.order:
            print(f"{c}: {self.char_count[ord(c)]}")


class HTree:
    """A HuffmanTree to be used to encode and decode messages. """
    def __init__(self, c=None, freq=0, p0=None, p1=None, order=None):
        self.char = c
        self.freq = freq
        self.p0 = p0
        self.p1 = p1
        self.order = order

    def print_tree(self, level=0, path: str = ""):
        """Print the tree in an easy to understand format. """
        for i in range(level):
            print("--", end='')
        print(f"Char: {self.char}, count: {self.freq}. Path: {path}")
        if self.p0:
            self.p0.print_tree(level + 1, path + "0")
        if self.p1:
            self.p1.print_tree(level + 1, path + "1")

    ## For a specified tree and character,
    ##  determine if the character is in the tree,
    ##  and if so, the frequency count to get to it.
    ## Returns -1 if the character is not in the tree
    def get_char_count(self, char):
        """Get the frequency count for a character in the tree. """
        if self.char == char:
            return self.freq
        if self.p0:
            count = self.p0.get_char_count(char)
            if count != -1:
                return count
        if self.p1:
            count = self.p1.get_char_count(char)
            if count != -1:
                return count
        return -1

    ## For a specified tree and character,
    ##  determine if the character is in the tree,
    ##  and if so, the path to get to it.
    ## Returns "" if the character is not in the tree
    def get_char_path(self, target, path=""):
        """Get the path to a given character in this tree."""
        if self.char == target:
            return path
        if self.p0:
            path0 = self.p0.get_char_path(target, path + "0")
            if path0:
                return path0
        if self.p1:
            path1 = self.p1.get_char_path(target, path + "1")
            if path1:
                return path1
        return ""

    ## Produces a serialized output of the tree, in the format:
    ## A0C1000D1001E1010F1011G1100H1101B111
    ## where it's [char][pathToChar][char][pathToChar]
    ## This is a LOW priority; this should be the last thing to implement
    def serialize(self, path: str = ""):
        """Write the tree into a string format to make it easy to save. """
        if self.char is not None:
            return f"{self.char}{path}"

        serialized_tree = ""
        if self.p0:
            serialized_tree += self.p0.serialize(path + "0")
        if self.p1:
            serialized_tree += self.p1.serialize(path + "1")
        return serialized_tree

    ## Assumes all 1s and 0s are bits;
    ## A0C1000D1001E1010F1011G1100H1101B111
    ## Builds a tree based on the provided serialized tree string
    ## Doesn't populate it with frequencies, just the chars
    def deserialize(self, tree_string):
        """Given a serialized tree string, make this tree represent it. """
        self.char = None
        self.freq = 0
        self.p0 = None
        self.p1 = None

        i = 0
        while i < len(tree_string):
            char = tree_string[i]
            i += 1
            path = ""
            while i < len(tree_string) and tree_string[i] in "01":
                path += tree_string[i]
                i += 1
            self.create_path(char, path)

    ## If the path exists, check if the char is the same and returns true/false
    ## If the path doesn't exist, creates the path and creates leaf node with the given char
    def create_path(self, char: str, path: str):
        """Populate a path to a node given the path. """
        current_node = self
        for bit in path:
            if bit == "0":
                if current_node.p0 is None:
                    current_node.p0 = HTree()
                current_node = current_node.p0
            elif bit == "1":
                if current_node.p1 is None:
                    current_node.p1 = HTree()
                current_node = current_node.p1

        if current_node.char is not None and current_node.char != char:
            raise False
        current_node.char = char
        return True


class LUT:
    """A LookUp Table to be used to store characters and their associated strings. """
    def __init__(self):
        self.representation = [""] * 256

    def print_lookup_table(self):
        """Print the table out in a nice to view manner"""
        for i in range(256):
            if self.representation[i]:
                print(f"{chr(i)}: {self.representation[i]}")

    ## Saves the path for a given char
    ## e.g. set_encoding('A', '10010')
    def set_encoding(self, char, path):
        """Save the encoding for a given character. """
        self.representation[ord(char)] = path

    ## Returns the path for a given char
    ## e.g. get_encoding('A') returns '10010'
    def get_encoding(self, char):
        """Return the encoding for a given character. """
        return self.representation[ord(char)]

    ## Given the root of a Huffman Tree, populate this lookup table.
    def populate_from_huffman_tree(self, htree_root: HTree):
        """Given a Huffman Tree, populate this LookupTable"""
        self.create_lookup_table_helper(htree_root, "", "")

    ## I found it helpful to have a function such as this to help
    ## traverse the HTree to populate the table.
    ## Feel free to ignore it if you'd like, or write something for yourself.
    def create_lookup_table_helper(self, node, path, which_step):
        """Helper function to populate this LUT from a HuffmanTree. """
        if node.char is not None:
            self.set_encoding(node.char, path)
        if node.p0:
            self.create_lookup_table_helper(node.p0, path + "0", 0)
        if node.p1:
            self.create_lookup_table_helper(node.p1, path + "1", 1)

class SecretMessage:
    """A class that holds an encoded message and the Huffman Tree that was used to create it. """
    def __init__(self, encoded_message: BitSeq, huffman_tree: HTree):
        self.encoded_bit_sequence = encoded_message
        self.huffman_tree = huffman_tree


## This is the function that actually creates the HuffmanTree.
## Follow the process outlined in the README,
## in the "Creating the mapping: The Huffman Tree" section.
def create_encoding_tree(char_counts: FreqTable) -> HTree:
    """Create an encoding tree to be used to encode the message. """
    priority_queue = []
    order_counter = 0  # Counter to track insertion order

    for i in char_counts.order:
        priority_queue.append(HTree(i, char_counts.get_char_count(i), order=order_counter))
        order_counter += 1

    # Sort the list by frequency
    priority_queue.sort(key=lambda node: (node.freq, node.order))

    # Build the tree
    while len(priority_queue) > 1:
        node0 = priority_queue.pop(0)
        node1 = priority_queue.pop(0)
        new_node = HTree(None, node0.freq + node1.freq, node0, node1, order=min(node0.order, node1.order))
        priority_queue.append(new_node)
        priority_queue.sort(key=lambda node: (node.freq, node.order))

    return priority_queue[0] if priority_queue else None


## The Encoder class is used to do encoding;
## It holds all the things we need to encode.
## This makes it helpful to inspect and test that
## all the pieces are working as expected.
class Encoder:
    """An Encoder encapsulates the entire process to create a SecretMessage via the use of a HuffmanTree. """
    def __init__(self):
        self.freq_table = None
        self.lookup_table = None
        self.huffman_tree = None
        self.encoded_bit_sequence = None

    ## Given a message,do all the steps to encode the message.
    ## When this is complete, the Encoder should have the
    ##  freq_table, lookup_table, huffman_tree, and encoded_bit_sequence
    ##  attributes should all be populated. (this allows us to test all the things)
    ## The huffman_tree and encoded_bit_sequence should be returned in a
    ##  SecretMessage object, so it can be "sent to a someone else".
    def encode(self, message_to_encode) -> SecretMessage:
        """Creates a SecretMessage from a raw message. """

        # Create a Frequency Table
        self.freq_table = FreqTable()
        self.freq_table.populate(message_to_encode)

        # Create a Huffman Tree
        self.huffman_tree = create_encoding_tree(self.freq_table)

        # Create a Lookup Table
        self.lookup_table = LUT()
        self.lookup_table.populate_from_huffman_tree(self.huffman_tree)

        # Encode the Message
        self.encoded_bit_sequence = BitSeq()
        encoded_message = ''.join(self.lookup_table.get_encoding(char) for char in message_to_encode)
        self.encoded_bit_sequence.pack_bits(encoded_message)

        # Return a SecretMessage
        return SecretMessage(self.encoded_bit_sequence, self.huffman_tree)


class Decoder:
    """A Decoder uses a Huffman Tree to decode a SecretMessage. """
    def __init__(self, huffman_tree: HTree):
        self.huffman_tree = huffman_tree

    ## Do the decoding of the provided message, using the
    ## self.huffman_tree.
    def decode(self, secret_message: BitSeq):
        """Decode the message, based on the HuffmanTree in this Decoder. """
        decoded_message = ""
        current_node = self.huffman_tree

        for bit_index in range(secret_message.num_bits_in_seq):
            # Get the next bit from the encoded bit sequence
            bit = secret_message.get_bit(bit_index)

            # Select the left or right node based on the bit value
            if bit == 0:
                current_node = current_node.p0
            else:
                current_node = current_node.p1

            # Check whether the leaf node is reached
            if current_node.char is not None:
                decoded_message += current_node.char
                # Reset to the root node of the tree
                current_node = self.huffman_tree

        return decoded_message