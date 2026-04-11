"""Invalid-action masking distributions derived from ``sb3_contrib``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import Distribution
from torch import nn
from torch.distributions import Categorical

SelfMaskableCategoricalDistribution = TypeVar(
    "SelfMaskableCategoricalDistribution",
    bound="MaskableCategoricalDistribution",
)
SelfMaskableMultiCategoricalDistribution = TypeVar(
    "SelfMaskableMultiCategoricalDistribution",
    bound="MaskableMultiCategoricalDistribution",
)
MaybeMasks = th.Tensor | np.ndarray | None


class MaskableCategorical(Categorical):
    """PyTorch categorical distribution with invalid-action masking support."""

    def __init__(
        self,
        probs: th.Tensor | None = None,
        logits: th.Tensor | None = None,
        validate_args: bool | None = None,
        masks: MaybeMasks = None,
    ) -> None:
        self.masks: th.Tensor | None = None
        super().__init__(probs=probs, logits=logits, validate_args=validate_args)
        self._original_logits = self.logits
        self.apply_masking(masks)

    def apply_masking(self, masks: MaybeMasks) -> None:
        """Mask out invalid actions by driving their logits to a huge negative."""
        if masks is None:
            self.masks = None
            logits = self._original_logits
        else:
            device = self.logits.device
            self.masks = th.as_tensor(masks, dtype=th.bool, device=device).reshape(
                self.logits.shape
            )
            huge_negative = th.tensor(-1e8, dtype=self.logits.dtype, device=device)
            logits = th.where(self.masks, self._original_logits, huge_negative)

        self.__dict__.pop("probs", None)
        super().__init__(logits=logits, validate_args=self._validate_args)

    def entropy(self) -> th.Tensor:
        if self.masks is None:
            return super().entropy()

        device = self.logits.device
        p_log_p = self.logits * self.probs
        p_log_p = th.where(self.masks, p_log_p, th.tensor(0.0, device=device))
        return -p_log_p.sum(-1)


class MaskableDistribution(Distribution, ABC):
    """Distribution interface that supports runtime action masking."""

    @abstractmethod
    def apply_masking(self, masks: MaybeMasks) -> None:
        """Mask out invalid actions in the current distribution instance."""

    @abstractmethod
    def proba_distribution_net(
        self,
        *args: object,
        **kwargs: object,
    ) -> nn.Module:
        """Create the network head for this distribution."""


class MaskableCategoricalDistribution(MaskableDistribution):
    """Categorical distribution for ``Discrete`` action spaces."""

    distribution: MaskableCategorical

    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, self.action_dim)

    def proba_distribution(
        self: SelfMaskableCategoricalDistribution,
        action_logits: th.Tensor,
    ) -> SelfMaskableCategoricalDistribution:
        reshaped_logits = action_logits.view(-1, self.action_dim)
        self.distribution = MaskableCategorical(logits=reshaped_logits)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> th.Tensor:
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        return th.argmax(self.distribution.probs, dim=1)

    def actions_from_params(
        self,
        action_logits: th.Tensor,
        deterministic: bool = False,
    ) -> th.Tensor:
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self,
        action_logits: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        return actions, self.log_prob(actions)

    def apply_masking(self, masks: MaybeMasks) -> None:
        self.distribution.apply_masking(masks)


class MaskableMultiCategoricalDistribution(MaskableDistribution):
    """Masked ``MultiDiscrete`` distribution."""

    def __init__(self, action_dims: list[int]) -> None:
        super().__init__()
        self.action_dims = action_dims
        self.distributions: list[MaskableCategorical] = []

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, sum(self.action_dims))

    def proba_distribution(
        self: SelfMaskableMultiCategoricalDistribution,
        action_logits: th.Tensor,
    ) -> SelfMaskableMultiCategoricalDistribution:
        reshaped_logits = action_logits.view(-1, sum(self.action_dims))
        self.distributions = [
            MaskableCategorical(logits=split)
            for split in th.split(reshaped_logits, self.action_dims, dim=1)
        ]
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        if not self.distributions:
            raise ValueError("Distribution parameters must be initialized first")

        reshaped_actions = actions.view(-1, len(self.action_dims))
        return th.stack(
            [
                distribution.log_prob(action)
                for distribution, action in zip(
                    self.distributions,
                    th.unbind(reshaped_actions, dim=1),
                    strict=True,
                )
            ],
            dim=1,
        ).sum(dim=1)

    def entropy(self) -> th.Tensor:
        if not self.distributions:
            raise ValueError("Distribution parameters must be initialized first")
        return th.stack(
            [distribution.entropy() for distribution in self.distributions],
            dim=1,
        ).sum(dim=1)

    def sample(self) -> th.Tensor:
        if not self.distributions:
            raise ValueError("Distribution parameters must be initialized first")
        return th.stack(
            [distribution.sample() for distribution in self.distributions],
            dim=1,
        )

    def mode(self) -> th.Tensor:
        if not self.distributions:
            raise ValueError("Distribution parameters must be initialized first")
        return th.stack(
            [
                th.argmax(distribution.probs, dim=1)
                for distribution in self.distributions
            ],
            dim=1,
        )

    def actions_from_params(
        self,
        action_logits: th.Tensor,
        deterministic: bool = False,
    ) -> th.Tensor:
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self,
        action_logits: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        return actions, self.log_prob(actions)

    def apply_masking(self, masks: MaybeMasks) -> None:
        if not self.distributions:
            raise ValueError("Distribution parameters must be initialized first")

        split_masks: list[th.Tensor | None] = [None] * len(self.distributions)
        if masks is not None:
            masks_tensor = th.as_tensor(masks).view(-1, sum(self.action_dims))
            split_masks = list(th.split(masks_tensor, self.action_dims, dim=1))

        for distribution, split_mask in zip(
            self.distributions,
            split_masks,
            strict=True,
        ):
            distribution.apply_masking(split_mask)


class MaskableBernoulliDistribution(MaskableMultiCategoricalDistribution):
    """Masked ``MultiBinary`` distribution."""

    def __init__(self, action_dim: int) -> None:
        super().__init__([2] * action_dim)


def make_masked_proba_distribution(action_space: spaces.Space) -> MaskableDistribution:
    """Build the appropriate masked distribution for a supported action space."""
    if isinstance(action_space, spaces.Discrete):
        return MaskableCategoricalDistribution(int(action_space.n))
    if isinstance(action_space, spaces.MultiDiscrete):
        return MaskableMultiCategoricalDistribution(
            [int(action_dim) for action_dim in action_space.nvec.ravel()]
        )
    if isinstance(action_space, spaces.MultiBinary):
        if not isinstance(action_space.n, int):
            raise NotImplementedError(
                "Multi-dimensional MultiBinary action spaces are not supported"
            )
        return MaskableBernoulliDistribution(action_space.n)

    raise NotImplementedError(
        "Masked probability distributions are only implemented for Discrete, "
        "MultiDiscrete, and MultiBinary action spaces."
    )
