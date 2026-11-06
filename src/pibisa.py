import dataclasses
import warnings
from typing import Literal

import numpy as np
import torch


class InfoLayer(torch.nn.Module):
    """This class represents the bottleneck to be trained.
    down_rate: The small mask will have dimensions input_shape/down_rate
    dist_range: The limits of an Uniform distribution U[-dist_range, dist_range]
    """

    def __init__(
        self, input_shape: tuple[int, int], down_rate: int, dist_range: int
    ) -> None:
        super().__init__()

        self.input_weight: int = input_shape[-1]
        self.input_height: int = input_shape[-2]

        self.mask_width: int = int(np.ceil(self.input_weight / down_rate))
        self.mask_height: int = int(np.ceil(self.input_height / down_rate))

        self.dist_range: int = dist_range

        self.small_mask = torch.empty(
            (self.mask_width, self.mask_height), dtype=torch.float32
        )
        self.small_mask = torch.nn.Parameter(self.small_mask)
        self.reset_small_mask()

    def forward(self, x):
        # Expanding dimensions is needed to match pytorch inputs
        mask_expanded = self.small_mask.expand(
            (1, 1, self.mask_width, self.mask_height)
        )
        mask_expanded = torch.nn.Upsample(
            size=(self.input_weight, self.input_height), mode="bicubic"
        )(mask_expanded)
        self.mask_final = torch.sigmoid(mask_expanded)

        return torch.mul(
            x, self.mask_final
        )  # returns the masked version of the input

    def reset_small_mask(self) -> None:
        torch.nn.init.uniform_(
            self.small_mask, -self.dist_range, self.dist_range
        )


class MaskLoss(torch.nn.Module):
    """
    Computes the term of the P-IBISA loss function that depends only on the mask.
    When the values of beta and phi are not specified (None), the parameters are
    set to 1.0 by default. Specifying only one of the parameters is not expected,
    so the class sets a default value for both and flags a warning.
    """

    def __init__(self, beta: float | None, phi: float | None) -> None:
        super().__init__()
        if beta is None or phi is None:
            warnings.warn("Setting both beta and phi to 1.0")
            beta, phi = 1.0, 1.0
        self.beta = beta
        self.phi = phi

    def forward(self, saliency_map):
        k = saliency_map.numel() * 3

        complexity_loss = self._total_norm(saliency_map)
        variation_loss = self._total_variation(saliency_map)

        return (self.beta * complexity_loss + self.phi * variation_loss) / k

    def _total_variation(self, saliency_map):
        saliency_map = torch.squeeze(saliency_map)
        x_diff = torch.abs(saliency_map[1:, :] - saliency_map[:-1, :])
        y_diff = torch.abs(saliency_map[:, 1:] - saliency_map[:, :-1])

        return torch.sum(x_diff) + torch.sum(y_diff)

    # It is a shallow function, but it will improve readability
    def _total_norm(self, saliency_map):
        return torch.sum(torch.abs(saliency_map))


class SimilarityLoss(torch.nn.Module):
    """Responsible for computing the (dis)similarities between the embeddings.
    Both the cosine simiarity and the mean squared error are supported.
    """

    def __init__(
        self,
        sim_func: Literal["cosine_sim", "mse"] = "cosine_sim",
        sigma: float = 1.0,
    ) -> None:

        super().__init__()
        self.sigma = sigma

        if sim_func == "cosine_sim":
            self._loss = torch.nn.CosineEmbeddingLoss()
            self._compute = self._cosine_forward
        else:
            self._loss = torch.nn.MSELoss()
            self._compute = self._mse_forward

    def _cosine_forward(
        self, query: torch.Tensor, target: torch.Tensor, sim_pos: torch.Tensor
    ):
        return self._loss(query, target, sim_pos)

    def _mse_forward(
        self, query: torch.Tensor, target: torch.Tensor, sim_pos: torch.Tensor
    ):
        loss = self._loss(query, target)
        if sim_pos == 1:
            return loss
        else:
            return 1 / (loss + 1e-8)

    def forward(
        self,
        query_emb: torch.Tensor,
        target_emb: torch.Tensor,
        sim_pos: torch.Tensor,
    ):
        """
        sim_pos: when equals to one, compute similarities. When equals
        to minus one, compute dissimilarities
        """
        return self.sigma * self._compute(
            query_emb, target_emb, sim_pos.to(target_emb.device)
        )


@dataclasses.dataclass
class PibisaConfig:
    number_masks: int = 20
    epochs: int = 15
    sim_func: Literal["cosine_sim", "mse"] = "cosine_sim"

    input_shape: tuple = (224, 224)
    down_rate: int = 32
    dist_range: int = 3

    sigma: int = 1
    beta: int | None = None
    phi: int | None = None

    device: str = "cuda"


class PibisaAtt:
    def __init__(self, encoder_forward, config: PibisaConfig) -> None:
        """This class receive the parameters for PIBISA the encoder forward method
        of the backbone we want to explain.
        """
        self.epochs = config.epochs
        self.number_masks = config.number_masks
        self.beta = config.beta
        self.phi = config.phi

        self.encoder_forward = encoder_forward
        self.bottleneck_layer = InfoLayer(
            config.input_shape, config.down_rate, config.dist_range
        ).to(config.device)
        self.loss_mask = MaskLoss(config.beta, config.phi).to(config.device)
        self.loss_sim = SimilarityLoss(config.sim_func, config.sigma).to(
            config.device
        )

        parameters = list(self.bottleneck_layer.parameters())
        if self.loss_mask.parameters() is not None:
            parameters.extend(list(self.loss_mask.parameters()))

        self.optimizer = torch.optim.Adam(parameters, lr=1)

    def compute_attribution(
        self,
        query_image: torch.Tensor,
        target_emb: torch.Tensor,
        pos_sim: int = 1,
    ) -> np.ndarray:
        """Computes the PIBISA attribution. Returns the saliency map as a normalized numpy array.
        target_emb: The target embedding that will be used to compute the attribution. It is the
        result of passing a target image/text through the backbone
        pos_sim: assumes the values of +1 if we want to compute similarities, or -1 if we want
        dissimilarities
        """
        pos_sim_tensor = torch.tensor([pos_sim], dtype=torch.long)
        self.final_att = np.zeros(
            (
                1,
                1,
                self.bottleneck_layer.input_height,
                self.bottleneck_layer.input_height,
            )
        )
        for _ in range(self.number_masks):
            self.bottleneck_layer.reset_small_mask()

            self.optimizer.state.clear()

            for _ in range(self.epochs):
                self.optimizer.zero_grad()
                query_emb = self.bottleneck_layer(query_image)
                query_emb = self.encoder_forward(query_emb)
                # target_emb is detached because its graph is not needed.
                # mask_final is only available after a forward pass,
                # and represent the current stage of the attribution.
                loss = self.loss_sim(
                    query_emb, target_emb.detach(), pos_sim_tensor
                ) + self.loss_mask(self.bottleneck_layer.mask_final)
                loss.backward(retain_graph=False)
                self.optimizer.step()

            self.final_att = (
                self.final_att
                + self.bottleneck_layer.mask_final.detach().cpu().numpy()
            )

        return self.final_att / self.number_masks
