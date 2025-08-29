import heapq
from typing import Dict, List, Tuple, Optional
import torch

class HuffmanNode:
    def __init__(self, symbol: Optional[int] = None, freq: int = 0):
        self.symbol = symbol
        self.freq = freq
        self.left: Optional['HuffmanNode'] = None
        self.right: Optional['HuffmanNode'] = None

    # Define comparison operators for heapq
    def __lt__(self, other: 'HuffmanNode') -> bool:
        return self.freq < other.freq
def huffman_encode(vector_mat_quantized_int: torch.Tensor) -> Tuple[torch.bool, Dict[int, str]]:
    """
    Encodes the input tensor using Huffman encoding and returns a bool tensor representing the bitstream.

    Args:
        vector_mat_quantized_int (torch.Tensor): Input tensor of integers.

    Returns:
        Tuple[torch.bool, Dict[int, str]]: Encoded data as a bool tensor and the Huffman codebook.
    """
    # Ensure the input tensor is on CPU for processing
    if vector_mat_quantized_int.is_cuda:
        vector_mat_quantized_int = vector_mat_quantized_int.cpu()

    # Flatten the tensor to 1D
    flat_tensor = vector_mat_quantized_int.view(-1)

    # Compute unique symbols and their counts using PyTorch for efficiency
    unique_symbols, counts = torch.unique(flat_tensor, return_counts=True)
    unique_symbols = unique_symbols.tolist()
    counts = counts.tolist()

    # Edge case: Empty input
    if len(unique_symbols) == 0:
        return torch.tensor([], dtype=torch.bool), {}

    # Edge case: Single unique symbol
    if len(unique_symbols) == 1:
        codebook = {unique_symbols[0]: '0'}
        encoded_bits = ['0'] * len(flat_tensor)
    else:
        # Build the Huffman tree using a min-heap
        heap = []
        for symbol, freq in zip(unique_symbols, counts):
            node = HuffmanNode(symbol, freq)
            heapq.heappush(heap, node)

        while len(heap) > 1:
            # Pop two nodes with the smallest frequencies
            node1 = heapq.heappop(heap)
            node2 = heapq.heappop(heap)

            # Merge these nodes
            merged = HuffmanNode()
            merged.freq = node1.freq + node2.freq
            merged.left = node1
            merged.right = node2

            # Push the merged node back into the heap
            heapq.heappush(heap, merged)

        # The remaining node is the root of the Huffman tree
        root = heap[0]

        # Generate Huffman codes by traversing the tree
        codebook = {}

        def generate_codes(node: HuffmanNode, current_code: str):
            if node is None:
                return
            if node.symbol is not None:
                codebook[node.symbol] = current_code or '0'  # Handle single symbol
                return
            generate_codes(node.left, current_code + '0')
            generate_codes(node.right, current_code + '1')

        generate_codes(root, '')

        # Encode the data using the generated Huffman codes
        # Use list comprehension for faster concatenation
        encoded_bits = [codebook[symbol] for symbol in flat_tensor.tolist()]

    # Concatenate all bitstrings into a single list of bits
    bit_list = []
    for bits in encoded_bits:
        bit_list.extend([bit == '1' for bit in bits])  # Convert '0'/'1' to False/True

    # Convert the list to a torch.bool tensor
    bit_tensor = torch.tensor(bit_list, dtype=torch.bool)

    return bit_tensor, codebook