# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Training-free multi-view fusion utilities for SAM 3D Objects.

Ported from facebookresearch/sam-3d-objects PR #37 (a multidiffusion
approach adapted from TRELLIS). The fusion math is intentionally identical
to the PR; only naming, comments, and logging were cleaned up.
"""
from contextlib import contextmanager
from typing import Literal

import torch
from loguru import logger


@contextmanager
def inject_generator_multi_view(
    generator,
    num_views: int,
    num_steps: int,
    mode: Literal["stochastic", "multidiffusion"] = "multidiffusion",
):
    """Temporarily patch ``generator._generate_dynamics`` for multi-view sampling.

    In ``multidiffusion`` mode every diffusion step runs the original
    dynamics once per view condition and averages the predictions, so one
    shared latent is guided by all views simultaneously. In ``stochastic``
    mode each step uses a single view's condition, rotating round-robin
    (cheaper, lower quality).

    The condition tokens are expected either stacked in a tensor of shape
    ``(num_views, ...)`` or as a list/tuple with one entry per view (see
    ``InferencePipeline.get_multi_view_condition_input``). The original
    dynamics function is restored on exit, including on exceptions.
    """
    original_dynamics = generator._generate_dynamics

    if mode == "stochastic":
        if num_views > num_steps:
            logger.warning(
                f"Number of views ({num_views}) exceeds number of steps "
                f"({num_steps}); some views will never be sampled."
            )

        cond_indices = (torch.arange(num_steps) % num_views).tolist()
        step_counter = [0]

        def _dynamics_stochastic(x_t, t, *args_conditionals, **kwargs_conditionals):
            cond_idx = cond_indices[step_counter[0] % len(cond_indices)]
            step_counter[0] += 1

            if len(args_conditionals) > 0:
                cond_tokens = args_conditionals[0]
                if isinstance(cond_tokens, (list, tuple)):
                    cond_i = (
                        cond_tokens[cond_idx : cond_idx + 1]
                        if isinstance(cond_tokens[0], torch.Tensor)
                        else [cond_tokens[cond_idx]]
                    )
                    new_args = (cond_i,) + args_conditionals[1:]
                elif (
                    isinstance(cond_tokens, torch.Tensor)
                    and cond_tokens.shape[0] == num_views
                ):
                    cond_i = cond_tokens[cond_idx : cond_idx + 1]
                    new_args = (cond_i,) + args_conditionals[1:]
                else:
                    new_args = args_conditionals
            else:
                new_args = args_conditionals

            return original_dynamics(x_t, t, *new_args, **kwargs_conditionals)

        generator._generate_dynamics = _dynamics_stochastic

    elif mode == "multidiffusion":

        def _dynamics_multidiffusion(x_t, t, *args_conditionals, **kwargs_conditionals):
            # Locate the condition tokens among the positional conditionals:
            # some generators prepend a scalar (e.g. a flag) before the tokens.
            cond_idx = 0
            if len(args_conditionals) > 0:
                first = args_conditionals[0]
                if isinstance(first, (int, float)) or (
                    isinstance(first, torch.Tensor) and first.numel() == 1
                ):
                    cond_idx = 1

            if len(args_conditionals) <= cond_idx:
                return original_dynamics(
                    x_t, t, *args_conditionals, **kwargs_conditionals
                )

            cond_tokens = args_conditionals[cond_idx]

            if isinstance(cond_tokens, (list, tuple)):
                view_conditions = cond_tokens
            elif (
                isinstance(cond_tokens, torch.Tensor)
                and cond_tokens.shape[0] == num_views
            ):
                view_conditions = [cond_tokens[i] for i in range(num_views)]
            else:
                logger.warning(
                    "Condition tokens are not organized per view; using the "
                    "same condition for every view."
                )
                view_conditions = [cond_tokens] * num_views

            preds = []
            for view_idx in range(num_views):
                view_cond = view_conditions[view_idx]
                new_args = (
                    args_conditionals[:cond_idx]
                    + (view_cond,)
                    + args_conditionals[cond_idx + 1 :]
                )
                preds.append(
                    original_dynamics(x_t, t, *new_args, **kwargs_conditionals)
                )

            if isinstance(preds[0], dict):
                return {
                    key: torch.stack([p[key] for p in preds]).mean(dim=0)
                    for key in preds[0].keys()
                }
            if isinstance(preds[0], (list, tuple)):
                return tuple(
                    torch.stack([p[i] for p in preds]).mean(dim=0)
                    for i in range(len(preds[0]))
                )
            return torch.stack(preds).mean(dim=0)

        generator._generate_dynamics = _dynamics_multidiffusion

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    try:
        yield
    finally:
        generator._generate_dynamics = original_dynamics
