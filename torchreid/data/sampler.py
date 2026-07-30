from __future__ import division, absolute_import
import copy
import numpy as np
import random
from collections import defaultdict
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler

AVAI_SAMPLERS = [
    'RandomIdentitySampler', 'RandomActionIdentitySampler',
    'SequentialSampler', 'RandomSampler', 'RandomDomainSampler',
    'RandomDatasetSampler'
]


class RandomIdentitySampler(Sampler):
    """Randomly samples N identities each with K instances.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid, dsetid).
        batch_size (int): batch size.
        num_instances (int): number of instances per identity in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        if batch_size < num_instances:
            raise ValueError(
                'batch_size={} must be no less '
                'than num_instances={}'.format(batch_size, num_instances)
            )

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, items in enumerate(data_source):
            pid = items[1]
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        assert len(self.pids) >= self.num_pids_per_batch

        # estimate number of examples in an epoch
        # TODO: improve precision
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(
                    idxs, size=self.num_instances, replace=True
                )
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomActionIdentitySampler(Sampler):
    """Samples several identities from each of several actions per batch.

    SoccerNet stores ``action_idx`` in tuple position 2 (``camid``). Each
    identity is used once per epoch. Identities with too few images are sampled
    with replacement; identities with more images use a random subset.
    """

    def __init__(
        self, data_source, batch_size, num_instances, num_actions
    ):
        if num_actions <= 0:
            raise ValueError('num_actions must be positive')
        if batch_size % (num_actions * num_instances) != 0:
            raise ValueError(
                'batch_size={} must be divisible by num_actions={} * '
                'num_instances={}'.format(
                    batch_size, num_actions, num_instances
                )
            )

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_actions = num_actions
        self.num_pids_per_action = (
            batch_size // (num_actions * num_instances)
        )

        self.action_pid_indices = defaultdict(lambda: defaultdict(list))
        for index, items in enumerate(data_source):
            pid = items[1]
            action_idx = items[2]
            self.action_pid_indices[action_idx][pid].append(index)

        self.actions = [
            action_idx
            for action_idx, pid_indices in self.action_pid_indices.items()
            if len(pid_indices) >= self.num_pids_per_action
        ]
        if len(self.actions) < self.num_actions:
            raise ValueError(
                'Need at least {} actions with {} identities each, but found '
                '{}'.format(
                    self.num_actions,
                    self.num_pids_per_action,
                    len(self.actions)
                )
            )

        block_counts = [
            len(self.action_pid_indices[action_idx])
            // self.num_pids_per_action
            for action_idx in self.actions
        ]
        self.num_batches = self._max_num_batches(block_counts)
        self.length = self.num_batches * self.batch_size

    def _max_num_batches(self, block_counts):
        """Maximum batches with at most one block per action in a batch."""
        upper_bound = sum(block_counts) // self.num_actions
        for num_batches in range(upper_bound, 0, -1):
            available_blocks = sum(
                min(count, num_batches) for count in block_counts
            )
            if available_blocks >= num_batches * self.num_actions:
                return num_batches
        return 0

    def _sample_instances(self, indices):
        if len(indices) >= self.num_instances:
            return random.sample(indices, self.num_instances)
        sampled = list(indices)
        sampled.extend(
            random.choices(
                indices, k=self.num_instances - len(indices)
            )
        )
        random.shuffle(sampled)
        return sampled

    def __iter__(self):
        action_blocks = {}
        for action_idx in self.actions:
            pids = list(self.action_pid_indices[action_idx].keys())
            random.shuffle(pids)
            blocks = []
            stop = len(pids) - self.num_pids_per_action + 1
            for start in range(0, stop, self.num_pids_per_action):
                selected_pids = pids[
                    start:start + self.num_pids_per_action
                ]
                block = []
                for pid in selected_pids:
                    block.extend(
                        self._sample_instances(
                            self.action_pid_indices[action_idx][pid]
                        )
                    )
                blocks.append(block)
            random.shuffle(blocks)
            action_blocks[action_idx] = blocks

        final_idxs = []
        for _ in range(self.num_batches):
            # Selecting actions with the most remaining blocks is a
            # Havel-Hakimi-style schedule that prevents incomplete batches.
            available_actions = [
                action_idx for action_idx, blocks in action_blocks.items()
                if blocks
            ]
            random.shuffle(available_actions)
            available_actions.sort(
                key=lambda action_idx: len(action_blocks[action_idx]),
                reverse=True
            )
            selected_actions = available_actions[:self.num_actions]
            if len(selected_actions) != self.num_actions:
                raise RuntimeError(
                    'Action-aware sampler could not construct a full batch'
                )
            for action_idx in selected_actions:
                final_idxs.extend(action_blocks[action_idx].pop())

        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomDomainSampler(Sampler):
    """Random domain sampler.

    We consider each camera as a visual domain.

    How does the sampling work:
    1. Randomly sample N cameras (based on the "camid" label).
    2. From each camera, randomly sample K images.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid, dsetid).
        batch_size (int): batch size.
        n_domain (int): number of cameras to sample in a batch.
    """

    def __init__(self, data_source, batch_size, n_domain):
        self.data_source = data_source

        # Keep track of image indices for each domain
        self.domain_dict = defaultdict(list)
        for i, items in enumerate(data_source):
            camid = items[2]
            self.domain_dict[camid].append(i)
        self.domains = list(self.domain_dict.keys())

        # Make sure each domain can be assigned an equal number of images
        if n_domain is None or n_domain <= 0:
            n_domain = len(self.domains)
        assert batch_size % n_domain == 0
        self.n_img_per_domain = batch_size // n_domain

        self.batch_size = batch_size
        self.n_domain = n_domain
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        domain_dict = copy.deepcopy(self.domain_dict)
        final_idxs = []
        stop_sampling = False

        while not stop_sampling:
            selected_domains = random.sample(self.domains, self.n_domain)

            for domain in selected_domains:
                idxs = domain_dict[domain]
                selected_idxs = random.sample(idxs, self.n_img_per_domain)
                final_idxs.extend(selected_idxs)

                for idx in selected_idxs:
                    domain_dict[domain].remove(idx)

                remaining = len(domain_dict[domain])
                if remaining < self.n_img_per_domain:
                    stop_sampling = True

        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomDatasetSampler(Sampler):
    """Random dataset sampler.

    How does the sampling work:
    1. Randomly sample N datasets (based on the "dsetid" label).
    2. From each dataset, randomly sample K images.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid, dsetid).
        batch_size (int): batch size.
        n_dataset (int): number of datasets to sample in a batch.
    """

    def __init__(self, data_source, batch_size, n_dataset):
        self.data_source = data_source

        # Keep track of image indices for each dataset
        self.dataset_dict = defaultdict(list)
        for i, items in enumerate(data_source):
            dsetid = items[3]
            self.dataset_dict[dsetid].append(i)
        self.datasets = list(self.dataset_dict.keys())

        # Make sure each dataset can be assigned an equal number of images
        if n_dataset is None or n_dataset <= 0:
            n_dataset = len(self.datasets)
        assert batch_size % n_dataset == 0
        self.n_img_per_dset = batch_size // n_dataset

        self.batch_size = batch_size
        self.n_dataset = n_dataset
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        dataset_dict = copy.deepcopy(self.dataset_dict)
        final_idxs = []
        stop_sampling = False

        while not stop_sampling:
            selected_datasets = random.sample(self.datasets, self.n_dataset)

            for dset in selected_datasets:
                idxs = dataset_dict[dset]
                selected_idxs = random.sample(idxs, self.n_img_per_dset)
                final_idxs.extend(selected_idxs)

                for idx in selected_idxs:
                    dataset_dict[dset].remove(idx)

                remaining = len(dataset_dict[dset])
                if remaining < self.n_img_per_dset:
                    stop_sampling = True

        return iter(final_idxs)

    def __len__(self):
        return self.length


def build_train_sampler(
    data_source,
    train_sampler,
    batch_size=32,
    num_instances=4,
    num_actions=1,
    num_cams=1,
    num_datasets=1,
    **kwargs
):
    """Builds a training sampler.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid).
        train_sampler (str): sampler name (default: ``RandomSampler``).
        batch_size (int, optional): batch size. Default is 32.
        num_instances (int, optional): number of instances per identity in a
            batch (when using an identity sampler). Default is 4.
        num_actions (int, optional): number of actions per batch (when using
            ``RandomActionIdentitySampler``). Default is 1.
        num_cams (int, optional): number of cameras to sample in a batch (when using
            ``RandomDomainSampler``). Default is 1.
        num_datasets (int, optional): number of datasets to sample in a batch (when
            using ``RandomDatasetSampler``). Default is 1.
    """
    assert train_sampler in AVAI_SAMPLERS, \
        'train_sampler must be one of {}, but got {}'.format(AVAI_SAMPLERS, train_sampler)

    if train_sampler == 'RandomIdentitySampler':
        sampler = RandomIdentitySampler(data_source, batch_size, num_instances)

    elif train_sampler == 'RandomActionIdentitySampler':
        sampler = RandomActionIdentitySampler(
            data_source, batch_size, num_instances, num_actions
        )

    elif train_sampler == 'RandomDomainSampler':
        sampler = RandomDomainSampler(data_source, batch_size, num_cams)

    elif train_sampler == 'RandomDatasetSampler':
        sampler = RandomDatasetSampler(data_source, batch_size, num_datasets)

    elif train_sampler == 'SequentialSampler':
        sampler = SequentialSampler(data_source)

    elif train_sampler == 'RandomSampler':
        sampler = RandomSampler(data_source)

    return sampler
