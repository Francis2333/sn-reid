from __future__ import absolute_import, division

import torch
from torch import nn

from .resnet import (
    Bottleneck,
    ResNet,
    init_pretrained_weights,
    model_urls,
)


__all__ = ['pcb_resnet50_p3']


class PCBResNet50P3(ResNet):
    """ResNet-50 with three horizontal part embeddings and one classifier."""

    def __init__(self, num_classes, loss='softmax', **kwargs):
        super(PCBResNet50P3, self).__init__(
            num_classes=num_classes,
            loss=loss,
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            last_stride=1,
            fc_dims=None,
            dropout_p=None,
            **kwargs
        )
        self.parts = 3
        self.part_dim = 256
        self.feature_dim = 512
        backbone_dim = 512 * Bottleneck.expansion

        self.parts_avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))
        self.part_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(backbone_dim, self.part_dim),
                    nn.BatchNorm1d(self.part_dim),
                    nn.ReLU(inplace=True),
                )
                for _ in range(self.parts)
            ]
        )
        self.embedding = nn.Sequential(
            nn.Linear(self.parts * self.part_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_head_params()

    def _init_head_params(self):
        modules = [
            self.part_projections,
            self.embedding,
            self.classifier,
        ]
        for module in modules:
            for layer in module.modules():
                if isinstance(layer, nn.BatchNorm1d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, 0, 0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        feature_map = self.featuremaps(x)
        pooled_parts = self.parts_avgpool(feature_map)

        part_embeddings = []
        for index, projection in enumerate(self.part_projections):
            part = pooled_parts[:, :, index, 0]
            part_embeddings.append(projection(part))

        embedding = self.embedding(
            torch.cat(part_embeddings, dim=1)
        )

        if not self.training:
            return embedding

        logits = self.classifier(embedding)
        if self.loss == 'softmax':
            return logits
        if self.loss == 'triplet':
            return logits, embedding
        raise KeyError('Unsupported loss: {}'.format(self.loss))


def pcb_resnet50_p3(
    num_classes,
    loss='softmax',
    pretrained=True,
    **kwargs
):
    model = PCBResNet50P3(
        num_classes=num_classes,
        loss=loss,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model
