import matplotlib.pyplot as plt
import numpy as np

def draw_one_accuracy_two_crs(accuracies, quant_cr, huffman_cr, title):
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot accuracies
    x_acc = np.arange(1, len(accuracies) + 1)
    ax1.plot(x_acc, accuracies, 'b-o', label='Accuracy')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(title)
    ax1.grid(True)
    ax1.legend()
    
    # Plot compression ratios
    x_cr = np.arange(1, len(quant_cr) + 1)
    ax2.plot(x_cr, quant_cr, 'r-o', label='Quantization CR')
    ax2.plot(x_cr, huffman_cr, 'g-o', label='Huffman CR')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Compression Ratio')
    ax2.set_title(title)
    ax2.grid(True)
    ax2.legend()
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{title}.png')
    # plt.show()
    # plt.close()
    