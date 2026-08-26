import numpy as np
import torch
import evaluate

from transformers import AutoProcessor
from tqdm import tqdm
from src.utils import unnormalize
from typing import Literal


class DeletionMetrics:
    """Computes the MoRF and LeRF metrics. It can observe either the logits in the case of
    classification; the cosine similarity in case of retrieval; or how the output of BLIP changes
    as we delete information for the input (nle mode).
    @encoderForward: the method used to compute the forward pass of the backbone. For CLIP it is
    model.encode_image or model.get_image_features. For BLIP, when evaluating NLEs, it is model.generate
    """

    def __init__(
        self,
        encoder_forward,
        observed_out: Literal["logit", "similarity", "nle"] = "logit",
        number_steps: int = 784,
        input_shape: tuple = (224, 224),
    ):

        self.encoder_forward = encoder_forward
        self.task = observed_out
        self.h, self.w = input_shape
        self.block_size = int(np.sqrt(self.h * self.w / number_steps))

        if observed_out == "nle":
            self.processor = AutoProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self.nlp_metrics = evaluate.combine(["bleu", "meteor", "rouge"])

        # Bind the right similarity function once
        self._compute_sim = {
            "logit": self._logit_sim,
            "similarity": self._cosine_sim,
            "nle": self._nle_sim,
        }[observed_out]

    def _logit_sim(self, input_image: torch.Tensor, target: torch.Tensor):
        """@target is the index that represents the original predicted class"""
        preds = self.encoder_forward(input_image)
        preds = torch.nn.functional.softmax(preds, dim=1)
        return preds[0, target].detach().cpu().numpy()

    def _cosine_sim(self, input_image: torch.Tensor, target: torch.Tensor):
        """@target is the embedding obtained by passing the target input through
        its backbone. The embedding can be obtained either from text, or from another
        image.
        """
        preds = self.encoder_forward(input_image)
        return (
            torch.nn.functional.cosine_similarity(target, preds)
            .detach()
            .cpu()
            .numpy()
        )

    def _nle_sim(self, input_image: torch.Tensor, target: str):
        """@target is the original text predicted by BLIP"""
        input_image = unnormalize(input_image)
        inputs = self.processor(input_image, return_tensors="pt").to("cuda")
        pred_text = self.encoder_forward(**inputs)
        pred_text = self.processor.decode(
            pred_text[0], skip_special_tokens=True
        )
        m = self.nlp_metrics.compute(
            predictions=[pred_text], references=[target]
        )
        return np.array(
            [m["meteor"], m["bleu"], m["rouge1"], m["rouge2"], m["rougeL"]]
        )

    def sort_coordinates(
        self, saliency_map: np.ndarray, reverse: bool = True
    ) -> list:
        """Takes a saliency map and return its sorted coordenates based on the most/least
        important pixels. When @reverse = True, sort by the most relevant first.
        """
        saliency_map = np.squeeze(saliency_map)
        assert (self.h, self.w) == saliency_map.shape, (
            "Saliency shape must match @inputShape"
        )

        coords = []
        importance = []
        for i in range(0, self.h, self.block_size):
            for j in range(0, self.w, self.block_size):
                importance.append(
                    saliency_map[
                        i : i + self.block_size, j : j + self.block_size
                    ].sum()
                )
                coords.append((i, j))

        indices = np.argsort(np.array(importance))
        if reverse:
            indices = indices[::-1]
        return [coords[i] for i in indices]

    def compute_score(
        self,
        image_input: torch.Tensor,
        target: torch.Tensor | str,
        saliency_map: np.typing.NDArray,
        metric: Literal["MoRF", "LeRF"] = "MoRF",
    ) -> tuple[np.ndarray, float]:

        coords_sorted = self.sort_coordinates(
            saliency_map, reverse=(metric == "MoRF")
        )
        scores = [self._compute_sim(image_input, target)]

        mask = torch.ones_like(image_input, dtype=torch.float32)
        for idx, idy in coords_sorted[:-1]:
            mask[
                :, :, idx : idx + self.block_size, idy : idy + self.block_size
            ] = 0
            masked = (image_input * mask).to(image_input.device)
            scores.append(self._compute_sim(masked, target))

        scores = np.array(scores, dtype=np.float32)
        scores = (scores - scores.min(axis=0)) / (
            scores.max(axis=0) - scores.min(axis=0)
        )

        x = np.linspace(0, 1, len(scores))
        area = np.trapezoid(np.squeeze(scores), x, axis=0)

        return scores, area


class RetrievalMetrics:
    """Computes the Confidence Increase and Confidence Drop metrics for the retrieval
    task. The class is initialized only witht the encoder forward method. The input image,
    target embedding and saliency map should be passed to the compute_increase_drop method.
    """

    def __init__(self, encoder_forward, percentile: int = 50) -> None:

        self.encoder_forward = encoder_forward
        self.percentile = percentile

    def _create_sal_masks(self, saliency_list: list) -> torch.Tensor:
        """Given a list of saliencies, returns binarized versions of it corresponding
        to the 50% of most important pixels
        """

        saliency_array = np.array(saliency_list)
        # Computes the 50% percentile for each of the saliencies
        threshold = np.percentile(
            saliency_array.reshape(len(saliency_array), -1),
            self.percentile,
            axis=-1,
        )

        # Defines a mapping matrix, containing the values of the 50% percentile for
        # each saliency.
        thresh_matrix = np.zeros_like(saliency_array)
        for j, th in enumerate(threshold):
            thresh_matrix[j, :, :] = th

        # Creates the binary mask using the threshold
        binary_mask = (saliency_array > thresh_matrix).astype(np.float32)
        binary_mask = torch.tensor(binary_mask)

        return torch.squeeze(binary_mask, dim=1)

    def confidence_increase_drop(
        self,
        input_query: torch.Tensor,
        input_targets,
        saliency_list: list[np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """For a given query and its target/saliencies pairs. Compute the
        Confidence Increase and Confidence Drop metrics for all targets.
        @inputTarget: list containing all target images for a given query
        @saliencyList: list containing the attributions highlights what elements in each
        target are similar to the query
        """
        input_targets = torch.squeeze(torch.stack(input_targets), dim=1).to(
            input_query.device
        )
        saliency_mask = self._create_sal_masks(saliency_list).to(
            input_query.device
        )

        query_emb = self.encoder_forward(input_query)
        target_embds = self.encoder_forward(input_targets)
        mask_embds = self.encoder_forward(input_targets * saliency_mask)

        similarity_original = torch.nn.functional.cosine_similarity(
            query_emb, target_embds
        )
        similarity_masked = torch.nn.functional.cosine_similarity(
            query_emb, mask_embds
        )

        diff = similarity_masked - similarity_original
        confidence_increase = (diff > 0).float()
        confidence_drop = torch.nn.functional.relu(-diff)
        confidence_drop = confidence_drop[confidence_drop > 0]

        return confidence_increase, confidence_drop
