#!/usr/bin/env python3

from functools import lru_cache
import torch
from torch.nn import Conv2d
from pathlib import Path

import train
from utils.train_utils import select_model


def initialize_model(model_name, model_load_path, n_classes, device, **kwargs):
    model = select_model(model_name, n_classes)
    assert model is not None
    state_dict = torch.load(model_load_path)
    model.load_state_dict(state_dict)
    model.cuda(device)
    model.train(False)

    # Embed forward hook into model for hidden feature extraction.
    _children = list(model.modules())
    feature_layer = next((_layer for _layer in reversed(_children) if isinstance(_layer, Conv2d)))

    def _hidden_feature_hook(layer, input, output):
        model.hidden_features = input[0].flatten(1)
    feature_layer.register_forward_hook(_hidden_feature_hook)
    return model


def initialize_dataloaders(class_encoding, batch_size, data_path, **kwargs):
    train_dl, val_dl = train.get_dataloader_keyword(data_path, class_list, class_encoding, batch_size, return_data_path=True)
    return train_dl, val_dl


def generate_embeddings(model, waveform):
    device = next(model.parameters()).device
    waveform = waveform.to(device)
    logits, features = model(waveform), model.hidden_features
    embeddings = torch.cat([logits, features], dim=1)
    return embeddings


@lru_cache
def _mkdir_if_not_exists(path):
    path.mkdir(parents=True, exist_ok=True)


def save_embeddings(data, paths):
    for concat_feat, path in zip(data, paths, strict=True):
        _mkdir_if_not_exists(path.parent)
        torch.save(concat_feat, path)


def process_data(dataloader, dataloader_name):
    assert dataloader_name in ["train", "test"]
    for waveform, _, data_path in dataloader:
        embeddings = generate_embeddings(model, waveform)
        data_names = map(lambda x :x.rsplit("/", 1)[-1].rsplit(".", 1)[0], data_path)
        data_paths = map(
            lambda name: Path(output_dir) / Path(f"{dataloader_name}/{name}.pt"),
            data_names
        )
        save_embeddings(embeddings, data_paths)


from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", dest="model_name", default="bcresnet")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--source", dest="model_load_path", default="model_save/weight/best.pt")
    parser.add_argument("--output", default="kws_embeddings")
    parser.add_argument("--data_path", default="dataset/kws")
    parser.add_argument("--batch", dest="batch_size", default=5, type=int)
    args = parser.parse_args()

    class_list = ["clipper", "irrigator", "pe", "scissors"]
    class_encoding = {category: index for index, category in enumerate(class_list)}

    train_dl, val_dl = initialize_dataloaders(class_encoding, **vars(args))
    model = initialize_model(**vars(args), n_classes=len(class_list))

    output_dir = Path(args.output)

    process_data(train_dl, "train")
    process_data(val_dl, "test")

