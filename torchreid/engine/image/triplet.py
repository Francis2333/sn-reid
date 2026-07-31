from __future__ import division, print_function, absolute_import

from torchreid import metrics
from torchreid.losses import TripletLoss, CrossEntropyLoss

from ..engine import Engine


class ImageTripletEngine(Engine):
    r"""Triplet-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        margin (float, optional): margin for triplet loss. Default is 0.3.
        weight_t (float, optional): weight for triplet loss. Default is 1.
        weight_x (float, optional): weight for softmax loss. Default is 1.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.
        action_aware (bool, optional): mine triplet positives and negatives
            only within the same action. Default is False.
        triplet_warmup_epochs (int, optional): number of initial epochs that
            optimize cross-entropy only. Default is 0.

    Examples::
        
        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32,
            num_instances=4,
            train_sampler='RandomIdentitySampler' # this is important
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='triplet'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageTripletEngine(
            datamanager, model, optimizer, margin=0.3,
            weight_t=0.7, weight_x=1, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-triplet-market1501',
            print_freq=10
        )
    """

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        margin=0.3,
        weight_t=1,
        weight_x=1,
        scheduler=None,
        use_gpu=True,
        label_smooth=True,
        action_aware=False,
        triplet_warmup_epochs=0
    ):
        super(ImageTripletEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        assert weight_t >= 0 and weight_x >= 0
        assert weight_t + weight_x > 0
        self.weight_t = weight_t
        self.weight_x = weight_x
        self.action_aware = action_aware
        if triplet_warmup_epochs < 0:
            raise ValueError('triplet_warmup_epochs must be non-negative')
        if triplet_warmup_epochs > 0 and weight_x == 0:
            raise ValueError(
                'Triplet warm-up requires a positive cross-entropy weight'
            )
        self.triplet_warmup_epochs = triplet_warmup_epochs

        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)
        action_ids = data['camid'] if self.action_aware else None

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()
            if action_ids is not None:
                action_ids = action_ids.cuda()

        outputs, features = self.model(imgs)

        loss = 0
        loss_summary = {}

        triplet_active = (
            self.weight_t > 0
            and self.epoch >= self.triplet_warmup_epochs
        )
        if triplet_active:
            if isinstance(features, (tuple, list)):
                loss_t = sum(
                    self.criterion_t(feature, pids, action_ids)
                    for feature in features
                ) / len(features)
            else:
                loss_t = self.criterion_t(features, pids, action_ids)
            loss += self.weight_t * loss_t
            loss_summary['loss_t'] = loss_t.item()
        elif self.weight_t > 0:
            loss_summary['loss_t'] = 0.

        if self.weight_x > 0:
            loss_x = self.compute_loss(self.criterion_x, outputs, pids)
            loss += self.weight_x * loss_x
            loss_summary['loss_x'] = loss_x.item()
            loss_summary['acc'] = metrics.accuracy(outputs, pids)[0].item()

        feature_for_stats = (
            features[0] if isinstance(features, (tuple, list)) else features
        )
        detached_features = feature_for_stats.detach().float()
        loss_summary['feat_std'] = (
            detached_features
            .var(dim=0, unbiased=False)
            .mean()
            .sqrt()
            .item()
        )

        assert loss_summary

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_summary
