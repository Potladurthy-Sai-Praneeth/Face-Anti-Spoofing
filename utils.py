import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_depth_maps(outputs, ground_truth, binary):
    index = {0: None, 1: None}
    sorted_labels_to_check = sorted(torch.unique(binary).cpu().numpy())
    
    for label in sorted_labels_to_check:
        if label not in [0, 1]:
            continue 
        
        matches = (binary == label).nonzero(as_tuple=True)[0]
        if len(matches) > 0:
            index[label] = matches[0].item()

    present_labels = [label for label, idx in index.items() if idx is not None]
    
    num_plots = len(present_labels) * 2

    if num_plots == 0:
        print("No samples with label 0 or 1 found in this batch to plot.")
        return

    fig, axes = plt.subplots(1, num_plots, figsize=(6 * len(present_labels), 5), squeeze=False)
    axes = axes.flatten() 

    ax_counter = 0
    for label in present_labels:
        data_idx = index[label]

        # Plot Prediction
        pred_map = outputs[data_idx].cpu().detach().squeeze().numpy()
        axes[ax_counter].imshow(pred_map)
        axes[ax_counter].set_title(f'Prediction (Label {label})')
        axes[ax_counter].axis('off')
        ax_counter += 1

        # Plot Ground Truth
        gt_map = ground_truth[data_idx].cpu().detach().squeeze().numpy()
        axes[ax_counter].imshow(gt_map)
        axes[ax_counter].set_title(f'Ground Truth (Label {label})')
        axes[ax_counter].axis('off')
        ax_counter += 1
        
    plt.tight_layout()
    plt.show()

def calculate_classification_accuracy(binary_outputs, binary_targets):
    """Calculate binary classification accuracy"""
    predictions = torch.argmax(binary_outputs, dim=1)
    correct = (predictions == binary_targets).float()
    if predictions.numel() == 0:
        return 0.0
    accuracy = correct.sum() / predictions.size(0)
    return accuracy.item()
