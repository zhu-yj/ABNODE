import numpy as np
import torch


def skew(v,device):
    """
    Computes the skew-symmetric matrix for a batch of vectors.
    Input:
        v (torch.Tensor): A tensor of shape (batch_size, 3) containing vectors.
        device (torch.device): The device on which to create the resulting tensor.

    Return:
        torch.Tensor: A tensor of shape (batch_size, 3, 3) containing skew-symmetric matrices.
    """
    bs=v.shape[0] # Get the batch size (number of vectors)
    skew_v=[] # Initialize list to store skew matrices
    for i in range(bs): 
        # Create the skew-symmetric matrix for each vector
        skew_v.append([[0,-v[i,2],v[i,1]],
                         [v[i,2],0,-v[i,0]],
                         [-v[i,1],v[i,0],0]])

    y=torch.tensor(skew_v,device=device)
    return y