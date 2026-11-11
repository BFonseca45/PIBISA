import sys

sys.path.insert(0, "/path_to_repo/PIBISA/src/")

import csv

import hydra
import numpy as np
import omegaconf
import torch
import torch.utils.data
import torchvision
import torchvision.datasets.imagenet
import tqdm

import src.metrics_pibisa
import src.pibisa
import src.utils


@hydra.main(
    version_base=None,
    config_path="./configs/",
    config_name="config_classification.yaml",
)
def run_main(cfg: omegaconf.DictConfig):
    csv_file, _ = src.utils.create_save_folder(
        cfg=cfg, metrics_list=["MoRF", "LeRF"]
    )

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    model, metric_model = src.utils.select_model_with_head(
        model_name=cfg.model
    )
    model = model.eval().to(device)
    metric_model = metric_model.eval().to(device)

    gen_cpu = torch.Generator(device="cpu")
    gen_cpu.manual_seed(45)

    dataset = torchvision.datasets.imagenet.ImageNet(
        root="/path_to_dataset/ImageNet/",
        split="val",
        transform=transform,
    )
    sub_set = torch.utils.data.Subset(dataset=dataset, indices=range(10000))
    data_loader = torch.utils.data.DataLoader(
        dataset=sub_set,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        generator=gen_cpu,
    )

    lerf = []
    morf = []

    del_metrics = src.metrics_pibisa.DeletionMetrics(
        encoder_forward=metric_model.forward
    )

    config = src.pibisa.PibisaConfig(beta=cfg.beta, phi=cfg.phi, device=device)
    pibisa = src.pibisa.PibisaAtt(model.forward, config)

    i = 0
    for x, _ in tqdm.tqdm(data_loader):
        x = x.to(device)

        saliency = pibisa.compute_attribution(
            query_image=x, target_emb=model.forward(x)
        )

        y_target = metric_model(x)
        y_target = torch.argmax(y_target, dim=1)

        morf_area, _ = del_metrics.compute_score(
            image_input=x,
            target=y_target,
            saliency_map=saliency,
            metric="MoRF",
        )

        lerf_area, _ = del_metrics.compute_score(
            image_input=x,
            target=y_target,
            saliency_map=saliency,
            metric="LeRF",
        )

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([morf_area, lerf_area])

        lerf.append(lerf_area)
        morf.append(morf_area)

        i = i + 1
        if not (i % 100):
            morf_mean = np.nanmean(morf)
            learf_mean = np.nanmean(lerf)

            print(morf_mean)
            print(learf_mean)


if __name__ == "__main__":
    run_main()
