import collections
import logging

import torch

import models


LOGGER = logging.getLogger(__name__)


def test_integration_1():
    d_a = 2
    d_b = 3
    d_c = 4

    fs2dim = collections.OrderedDict({
        "ab": (d_a, d_b),
        "bc": (d_b, d_c),
        "ca": (d_c, d_a)
    })
    data = collections.OrderedDict({
        "ab": torch.tensor([  # shape = 2 * 3 = 6
            1, 2, 3, 1, 5, 4
        ]),
        "bc": torch.tensor([  # shape = 3 * 4 = 12
            3, 4, 3, 2, 1, 6
        ]),
        "ca": torch.tensor([  # shape = 4 * 2 = 8
            7, 7, 7, 1, 3, 4
        ])
    })

    losses = models.core.mle_train(
        models.core.factor_model,
        (fs2dim,),
        dict(data=data),
    )