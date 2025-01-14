import numpy as np
import torch
from advertorch.attacks import Attack, LabelMixin
from advertorch.attacks.utils import rand_init_delta
from advertorch.utils import batch_multiply, batch_clamp
from advertorch.utils import clamp

from torch import nn


class IAE_FGSM(Attack, LabelMixin):
    """
    One step fast gradient sign method (Goodfellow et al, 2014).

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: attack step size.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: indicate if this is a targeted attack.
    """

    def __init__(self, predict, loss_fn=None, eps=0.3, eps_iter=4/255, clip_min=0.,
                 clip_max=1., targeted=False):
        """
        Create an instance of the GradientSignAttack.
        """
        super(IAE_FGSM, self).__init__(
            predict, loss_fn, clip_min, clip_max)

        self.eps = eps
        self.eps_iter = eps_iter
        self.targeted = targeted
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_l1 = nn.L1Loss()
        self.loss_mse = nn.MSELoss()
        self.loss_kl = nn.KLDivLoss(reduction='batchmean')

    def perturb(self, x, y=None, num_iter=1):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """

        x, y = self._verify_and_process_inputs(x, y)
        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        rand_init_delta(delta, x, ord=np.inf, eps=self.eps, clip_min=self.clip_min, clip_max=self.clip_max)
        delta.requires_grad_()

        for _ in range(num_iter):
            xadv = x + delta
            ori_features, ori_outputs = self.predict.extract_feature(x)
            features, outputs = self.predict.extract_feature(xadv)
            temperature = 10.
            loss1 = self.loss_ce(outputs/temperature, y)
            loss2 = self.loss_kl(torch.log_softmax(xadv, dim=1), torch.softmax(x, dim=1))
            loss3 = self.loss_kl(torch.log_softmax(features[-1], dim=1), torch.softmax(ori_features[-1].detach(), dim=1))
            lambda_3 = 0.1
            loss = -loss1 + lambda_3*loss2 - lambda_3*loss3
            loss.backward()
            grad_sign = delta.grad.detach().sign()
            delta.data = delta.data + batch_multiply(self.eps_iter, grad_sign)
            delta.data = batch_clamp(self.eps, delta.data)
            delta.data = clamp(x.data + delta.data, self.clip_min, self.clip_max) - x.data

        xadv = clamp(x + delta, self.clip_min, self.clip_max)

        return xadv.detach()
