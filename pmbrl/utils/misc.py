import torch

def one_hot(labels, num_classes, batch=False):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
        if not batch
            (tensor) encoded labels, sized [N, #classes].
        else
            (tensor) encoded labels, sized [dim(0)...dim(N-1), #classes].
    """
    y = torch.eye(num_classes)
    if not batch:
        return y[labels]
    else:
        return y[labels].squeeze(-2)