import pickle
import warnings

import clip
import hydra
import matplotlib.pyplot as plt
import torch
import torchvision
import tqdm
import wandb

warnings.filterwarnings("ignore")

from omegaconf import DictConfig
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    CLIPTokenizerFast,
)

from metrics_pibisa import *
from NIB.saliency import (
    chefer,
    fast_ig,
    gradcam,
    m2ib,
    mfaba,
    nib,
    rise,
    saliencymap,
)
from src.grad_eclip import *
from utils import *
import src.pibisa

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

blip_processor = AutoProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
blip = AutoModelForVision2Seq.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)


@hydra.main(
    version_base=None, config_path="./configs/", config_name="config_CLIP.yaml"
)
def run_main(cfg: DictConfig):
    warnings.filterwarnings("ignore")
    # Passing config to variables
    beta = cfg.param.beta
    server = cfg.run.server
    method = cfg.run.method

    wandb_run = wandb.init(
        project="PIBISA",
        config={
            "beta": beta,
            "server": server,
            "method": method,
        },
    )

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("Device in use: ", device)
    torch.manual_seed(0)

    eval_tool = DeletionMetrics(encoder_forward=blip, observed_out="nle")

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    dset = torchvision.datasets.CocoCaptions(
        root="/path_to_data/coco/val2017/",
        annFile="/path_to_data/coco/annotations/captions_val2017.json",
        transform=transform,
    )
    subset = torch.utils.data.Subset(dataset=dset, indices=range(1000))

    gen_cpu = torch.Generator(device="cpu")
    gen_cpu.manual_seed(45)
    dloader = torch.utils.data.DataLoader(
        dataset=subset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=6,
        generator=gen_cpu,
    )

    print("## Everything loaded, starting to run ##")

    i = 1
    MoRF = []
    LeRF = []
    captions = []

    savefile = "./out/metrics/CLIP_metrics_" + method + ".pkl"

    if method == "gradeclip":
        # Get model, processor, and tokenizer
        model, _ = clip.load("ViT-B/32", device=device)
        model.get_image_features = model.encode_image  # type:ignore
        model.get_text_features = model.encode_text  # type:ignore

        processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", use_fast=True
        )
    else:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            device  # type:ignore
        )
        processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        tokenizer = CLIPTokenizerFast.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    pibisa_cfg = src.pibisa.PibisaConfig(beta=beta, phi=1, device=device)
    pibisa = src.pibisa.PibisaAtt(
        encoder_forward=model.get_image_features, config=pibisa_cfg
    )

    for x, _ in tqdm.tqdm(dloader):
        # Generating the caption for a given image
        x_pil = unnormalize(x)
        x = processor(images=x_pil, return_tensors="pt")["pixel_values"].to(
            device
        )
        x = x.to(device)

        blip_inputs = blip_processor(x_pil, return_tensors="pt").to("cuda")
        blip_out = blip.generate(**blip_inputs)
        target_caption = blip_processor.decode(
            blip_out[0], skip_special_tokens=True
        )

        # Generating the CLIP text embedding
        text = clip.tokenize(target_caption).to(device)
        y_target = model.get_text_features(text).to(device)

        # Computing the saliency map to explain the caption
        # This could be joined into a selector function
        if method == "Random":
            sal = 3 * torch.rand((1, 1, 7, 7))
            sal = torch.nn.Upsample(size=(224, 224), mode="bicubic")(sal)
            sal = torch.sigmoid(sal)
            sal = sal.detach().cpu().numpy()
        elif method == "PIBISA":
            sal = pibisa.compute_attribution(
                query_image=x, target_emb=y_target
            )
        elif method == "PIBISA_ENC":
            sal = pibisa.compute_attribution(
                query_image=x, target_emb=model.get_image_features(x)
            )
        elif method == "M2IB":
            sal = generate_plot(
                x_pil,
                target_caption,
                m2ib,
                tokenizer=tokenizer,
                model=model,
                processor=processor,
            )
        elif method == "NIB":
            sal = generate_plot(
                x_pil,
                target_caption,
                nib,
                tokenizer=tokenizer,
                model=model,
                processor=processor,
            )
        elif method == "gradeclip":
            tf = model.encode_text(text)
            outputs, last_feat, vs, qs, ks, attns, atten_outs, map_size = (
                clip_encode_dense(x, n=1, clipmodel=model)
            )
            img_embedding = F.normalize(outputs[:, 0], dim=-1)
            tf = F.normalize(tf, dim=-1)

            cosine = (img_embedding @ tf.T)[0]
            sal = grad_eclip(cosine, qs, ks, vs, atten_outs, map_size)
            # clean up everything
            del outputs, last_feat, vs, qs, ks, attns, atten_outs, map_size
            torch.cuda.empty_cache()
            sal -= sal.min()
            sal /= sal.max()

            h, w = x.shape[-2:]
            resize = T.Resize((w, h))
            sal = resize(sal.unsqueeze(0))
            sal = sal.detach().cpu().numpy()
        else:
            raise TypeError("Insert a valid attribution method")

        _, morf = eval_tool.compute_score(
            image_input=x,
            target=target_caption,
            saliency_map=sal.squeeze(),
            metric="MoRF",
        )
        _, lerf = eval_tool.compute_score(
            image_input=x,
            target=target_caption,
            saliency_map=sal.squeeze(),
            metric="LeRF",
        )

        MoRF.append(morf)
        LeRF.append(lerf)

        fig_savefile = "./out/clip_images/saliency" + str(i) + method + ".png"
        plt.imshow(x_pil)
        plt.imshow(np.squeeze(sal), cmap="turbo", alpha=0.5)
        plt.title("Image n " + str(i))
        plt.axis("off")

        plt.savefig(fig_savefile)
        plt.close()

        if not (i % 5):
            captions.append(target_caption)

            tosave = {"MoRF": MoRF, "LeRF": LeRF, "Captions": captions}
            with open(savefile, "wb") as open_file:
                pickle.dump(tosave, open_file)

            MoRF_mean = np.nanmean(MoRF, axis=0)
            LeRF_mean = np.nanmean(LeRF, axis=0)

            wandb_run.log(
                {
                    "Meteor_MoRF": MoRF_mean[0],
                    "Bleu_MoRF": MoRF_mean[1],
                    "Rouge1_MoRF": MoRF_mean[2],
                    "Rouge2_MoRF": MoRF_mean[3],
                    "RougeL_MoRF": MoRF_mean[4],
                }
            )
            wandb_run.log(
                {
                    "Meteor_LeRF": LeRF_mean[0],
                    "Bleu_LeRF": LeRF_mean[1],
                    "Rouge1_LeRF": LeRF_mean[2],
                    "Rouge2_LeRF": LeRF_mean[3],
                    "RougeL_LeRF": LeRF_mean[4],
                }
            )
        i += 1


def generate_plot(image, text, method, model, tokenizer, processor):
    image_feat = processor(images=image, return_tensors="pt")[
        "pixel_values"
    ].to(device)
    image_features = model.get_image_features(image_feat)
    text_ids = torch.tensor(
        [tokenizer.encode(text, add_special_tokens=True)]
    ).to(device)
    text_words = tokenizer.convert_ids_to_tokens(text_ids[0].tolist())
    text_features = model.get_text_features(text_ids)

    if method in [chefer, fast_ig, gradcam, mfaba, saliencymap]:
        vmap, tmap = method(model, processor, [text], [image])
    elif method == rise:
        vmap, tmap = rise(
            model, image_feat, [text_ids], image_features, text_features
        )
        tmap = [tmap[0].detach().cpu().numpy()]
    elif method == m2ib:
        vmap, tmap = m2ib(model, [text_ids], image_feat, 0.1)
    elif method == nib:
        vmap, tmap = nib(model, [text_ids], image_feat, 10, 9)
    else:
        raise TypeError("Choose a valid attribution method")

    if vmap.shape[1] == 3:
        vmap = vmap.mean(1)

    return np.squeeze(vmap[0])


if __name__ == "__main__":
    run_main()
