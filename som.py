"""A fast implementation of 2D-SOM in PyTorch."""

import torch


def sed(x, y):
    """Compute the squared Euclidean distance between x and y."""
    expanded_x = x.unsqueeze(dim=1)
    batchwise_difference = y - expanded_x
    differences_raised = torch.pow(batchwise_difference, 2)
    distances = torch.sum(differences_raised, axis=2)
    return distances


def wtac(distances, labels):
    """Winner Takes-All Competition."""
    winning_indices = torch.min(distances, dim=1).indices
    winning_labels = labels[winning_indices].squeeze()
    return winning_labels


class SOM2D(torch.nn.Module):
    """2D-Self Organizing Map."""
    def __init__(self, shape, input_dim, alpha=0.3, sigma=None):
        super().__init__()
        self.shape = shape
        self.input_dim = input_dim
        self.alpha = alpha
        self.sigma = sigma or max(*shape) / 2.0

        size = shape[0] * shape[1]
        x = torch.arange(shape[0])
        y = torch.arange(shape[1])
        self.weights = torch.rand(size, input_dim)
        self.grid = torch.stack(torch.meshgrid(x, y), dim=-1).reshape(-1, 2)

    def competition(self, inputs):
        d = sed(inputs, self.weights)
        return wtac(d, self.grid)

    def fit(self, inputs, epochs=1, lr=1.0, verbose=True):
        """Run the unsupervised SOM training algorithm."""
        for i in range(epochs):
            if verbose:
                print(f'Epoch {i+1}/{epochs}')

            # compute winning neurons
            d = sed(inputs, self.weights)
            wplocs = wtac(d, self.grid)

            # learning rate decay
            lrt = lr - (i / epochs)
            alphat = lrt * self.alpha
            sigmat = lrt * self.sigma

            # compute delta for the entire batch
            gd = sed(wplocs, self.grid)
            neighborhood = torch.exp(-gd / sigmat**2)
            diff = inputs.unsqueeze(dim=1) - self.weights
            delta = lrt * alphat * neighborhood.unsqueeze(dim=-1) * diff

            # sum up all the changes for the batch
            delta = torch.sum(delta, dim=0)

            # perform update
            self.weights = self.weights + delta
