import unittest
from collections import Counter, defaultdict

import torch

from torchreid.data.sampler import RandomActionIdentitySampler
from torchreid.losses import TripletLoss


class RandomActionIdentitySamplerTest(unittest.TestCase):

    def setUp(self):
        self.data = []
        for action_idx in range(4):
            for local_pid in range(4):
                pid = action_idx * 10 + local_pid
                for image_idx in range(2):
                    self.data.append(
                        (
                            'a{}-p{}-i{}.png'.format(
                                action_idx, pid, image_idx
                            ),
                            pid,
                            action_idx,
                            0,
                        )
                    )

    def test_batch_layout(self):
        batch_size = 32
        sampler = RandomActionIdentitySampler(
            self.data,
            batch_size=batch_size,
            num_instances=4,
            num_actions=2,
        )
        sampled_indices = list(iter(sampler))

        self.assertEqual(len(sampled_indices), len(sampler))
        self.assertEqual(len(sampled_indices) % batch_size, 0)

        for start in range(0, len(sampled_indices), batch_size):
            batch = [
                self.data[index]
                for index in sampled_indices[start:start + batch_size]
            ]
            by_action = defaultdict(list)
            for item in batch:
                by_action[item[2]].append(item[1])

            self.assertEqual(len(by_action), 2)
            for pids in by_action.values():
                pid_counts = Counter(pids)
                self.assertEqual(len(pid_counts), 4)
                self.assertEqual(set(pid_counts.values()), {4})

    def test_rejects_incompatible_batch_shape(self):
        with self.assertRaises(ValueError):
            RandomActionIdentitySampler(
                self.data,
                batch_size=30,
                num_instances=4,
                num_actions=2,
            )


class ActionAwareTripletLossTest(unittest.TestCase):

    def test_ignores_cross_action_false_negatives(self):
        features = torch.tensor(
            [[0.0], [0.0], [10.0], [10.0], [0.1], [0.1], [5.0], [5.0]]
        )
        pids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        action_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        criterion = TripletLoss(margin=0.3)

        global_loss = criterion(features, pids)
        action_loss = criterion(features, pids, action_ids)

        self.assertLess(action_loss.item(), global_loss.item())


if __name__ == '__main__':
    unittest.main()
