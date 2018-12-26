import torch
from torchtext.data import BucketIterator


def build(dataset, batch_size=64, device='cpu', is_train=True):
	return BucketIterator(
        dataset=dataset,
        batch_size=batch_size,
        repeat=False,

        # sorts the data within each minibatch in decreasing order according
        # set to true if you want use pack_padded_sequences
        sort_key=dataset.sort_key,
        sort=is_train,
        sort_within_batch=is_train,
        # shuffle batches
        shuffle=is_train,
        device=torch.device(device),
        train=is_train
    )
