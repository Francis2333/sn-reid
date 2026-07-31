import os
import sys
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from torchreid.engine import ImageTripletEngine
from torchreid.metrics.rank import eval_soccernetv3
from torchreid.models import build_model


class SoccerNetMetricTest(unittest.TestCase):

    def test_cmc_is_padded_monotonically_for_short_actions(self):
        distmat = np.asarray(
            [
                [0.1, 9.0, 9.0, 9.0],
                [9.0, 0.1, 0.2, 0.3],
            ],
            dtype=np.float32,
        )
        q_pids = np.asarray([10, 20])
        g_pids = np.asarray([10, 30, 20, 40])
        q_actions = np.asarray([0, 1])
        g_actions = np.asarray([0, 1, 1, 1])

        cmc, mean_ap = eval_soccernetv3(
            distmat,
            q_pids,
            g_pids,
            q_actions,
            g_actions,
            max_rank=3,
        )

        np.testing.assert_allclose(cmc, [0.5, 1.0, 1.0])
        self.assertAlmostEqual(mean_ap, 0.75)
        self.assertTrue(np.all(np.diff(cmc) >= 0))


class ResNetEmbeddingHeadTest(unittest.TestCase):

    def test_fc512_embedding_has_no_final_relu(self):
        model = build_model(
            'resnet50_fc512',
            num_classes=4,
            loss='triplet',
            pretrained=False,
        )

        self.assertIsInstance(model.fc[-1], nn.Identity)


class _DummyDataManager:
    num_train_pids = 2
    train_loader = []
    test_loader = {}


class _DummyTripletModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(2, 2)
        self.classifier = nn.Linear(2, 2)

    def forward(self, inputs):
        features = self.embedding(inputs)
        return self.classifier(features), features


class TripletWarmupTest(unittest.TestCase):

    def test_warmup_skips_triplet_and_reports_feature_spread(self):
        model = _DummyTripletModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        engine = ImageTripletEngine(
            _DummyDataManager(),
            model,
            optimizer,
            weight_t=0.2,
            weight_x=1.0,
            use_gpu=False,
            action_aware=True,
            triplet_warmup_epochs=2,
        )
        engine.criterion_t = mock.Mock(
            side_effect=AssertionError('triplet loss ran during warm-up')
        )
        engine.epoch = 0

        summary = engine.forward_backward(
            {
                'img': torch.tensor(
                    [[0.0, 1.0], [1.0, 0.0], [0.5, 1.0], [1.0, 0.5]]
                ),
                'pid': torch.tensor([0, 1, 0, 1]),
                'camid': torch.tensor([0, 0, 0, 0]),
            }
        )

        self.assertEqual(summary['loss_t'], 0.0)
        self.assertGreater(summary['feat_std'], 0.0)
        engine.criterion_t.assert_not_called()


class BaselineEngineWiringTest(unittest.TestCase):

    def test_build_engine_forwards_action_aware_and_warmup(self):
        baseline_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'benchmarks', 'baseline')
        )
        sys.path.insert(0, baseline_dir)
        try:
            import main as baseline_main
            from default_config import get_default_config

            cfg = get_default_config()
            cfg.use_gpu = False
            cfg.data.type = 'image'
            cfg.loss.name = 'triplet'
            cfg.loss.triplet.action_aware = True
            cfg.loss.triplet.warmup_epochs = 5

            with mock.patch.object(
                baseline_main.torchreid.engine,
                'ImageTripletEngine',
            ) as engine_cls:
                baseline_main.build_engine(
                    cfg,
                    datamanager=mock.sentinel.datamanager,
                    model=mock.sentinel.model,
                    optimizer=mock.sentinel.optimizer,
                    scheduler=mock.sentinel.scheduler,
                )

            kwargs = engine_cls.call_args.kwargs
            self.assertTrue(kwargs['action_aware'])
            self.assertEqual(kwargs['triplet_warmup_epochs'], 5)
        finally:
            sys.path.remove(baseline_dir)


if __name__ == '__main__':
    unittest.main()
