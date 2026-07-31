"""A deliberately small AdamW core subset using DirectML-compatible operators."""

from __future__ import annotations

import math

import torch


class DirectMLCompatibleAdamWCore(torch.optim.Optimizer):
    """AdamW core equations without the wider torch.optim.AdamW API surface.

    This subset intentionally omits AMSGrad, maximize, capturable,
    differentiable, fused, and foreach modes. It supports dense gradients and
    a closure, which is evaluated with gradient recording enabled.
    """

    def __init__(
        self,
        params: object,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0.0:
            raise ValueError("learning rate must be nonnegative")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("betas must be in [0, 1)")
        if eps < 0.0:
            raise ValueError("epsilon must be nonnegative")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be nonnegative")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: object = None) -> object:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("DirectMLCompatibleAdamWCore does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                denominator = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(group["eps"])
                parameter.addcdiv_(exp_avg, denominator, value=-group["lr"] / bias_correction1)
        return loss
