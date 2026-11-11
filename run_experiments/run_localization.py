import sys

sys.path.insert(0, "/path_to_repo/PIBISA/src/")

import csv

import hydra
import numpy as np
import omegaconf
import torch
import torch.utils.data
import torchvision
import tqdm

import src.data
import src.metrics_quantus
import src.pibisa
import src.utils


@hydra.main(
    version_base=None,
    config_path="./configs/",
    config_name="config_localization.yaml",
)
def run_main(cfg: omegaconf.DictConfig):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    metrics_names = [
        "PointingGame",
        "AttributionLocalisation",
        "TopKIntersection",
        "RelevanceRankAccuracy",
    ]

    csv_file, _ = src.utils.create_save_folder(
        cfg=cfg, metrics_list=metrics_names
    )

    torch.manual_seed(0)
    gen_cpu = torch.Generator(device="cpu")
    gen_cpu.manual_seed(45)
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    path = "/path_to_data/ImageNetS300"
    split = "validation"

    dset = src.data.ImageNETs(path=path, split=split, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset=dset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        generator=gen_cpu,
    )

    config = src.pibisa.PibisaConfig(beta=cfg.beta, phi=cfg.phi, device=device)

    for architecture in cfg.model:
        model = src.utils.select_model_no_head(model_name=architecture)
        model = model.eval().to(device)

        if model._get_name() == "CLIP":
            pibisa = src.pibisa.PibisaAtt(
                encoder_forward=model.encode_image, config=config
            )
        else:
            pibisa = src.pibisa.PibisaAtt(
                encoder_forward=model.forward, config=config
            )

        total_metric = []
        for x, m in tqdm.tqdm(dataloader):
            x = x.to(device)

            if model._get_name() == "CLIP":
                saliency = pibisa.compute_attribution(
                    query_image=x,
                    target_emb=model.encode_image(x),  # type:ignore
                )
            else:
                saliency = pibisa.compute_attribution(
                    query_image=x, target_emb=model.forward(x)
                )

            current_metric = []
            for metric in metrics_names:
                metricObj = src.metrics_quantus.QuantusLocalisation(
                    metric_name=metric
                )
                current_metric.append(
                    metricObj.compute_metric(
                        model=None,
                        x_batch=x.detach().cpu().numpy(),
                        y_batch=0,
                        a_batch=saliency[None, ...],
                        s_batch=m.unsqueeze(0).numpy(),
                        device=device,
                    )
                )
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(current_metric)
            total_metric.append(current_metric)

        total_metric_np = np.array(total_metric)
        print("Architecture: ", architecture)
        print("Metrics:", np.mean(total_metric_np, axis=0))


if __name__ == "__main__":
    run_main()
