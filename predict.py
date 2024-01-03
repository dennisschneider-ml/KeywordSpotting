#!/usr/bin/env python3

import torch
import train
from torch.nn import Conv2d
from utils.train_utils import select_model


if __name__ == "__main__":
    model_name = "bcresnet"
    data_path = "dataset/kws"
    class_list = ["clipper", "irrigator", "pe", "scissors"]
    class_encoding = {category: index for index, category in enumerate(class_list)}
    batch_size = 5
    device = "cuda:2"

    model = select_model(model_name, len(class_list))
    state_dict = torch.load("model_save/weight/best.pt")
    model.load_state_dict(state_dict)
    train_dl, val_dl = train.get_dataloader_keyword(data_path, class_list, class_encoding, batch_size)
    model.cuda(device)
    model.train(False)

    # Embed forward hook into model for hidden feature extraction.
    _children = list(model.children())
    feature_layer = next((_layer for _layer in reversed(_children) if isinstance(_layer, Conv2d)))

    def _hidden_feature_hook(layer, input, output):
        model.hidden_features = input[0].flatten(1)
    feature_layer.register_forward_hook(_hidden_feature_hook)

    for waveform, _ in train_dl:
        waveform = waveform.to(device)
        logits, features = model(waveform), model.hidden_features
        concat_features = torch.cat([logits, features], dim=1)
        exit()

    for waveform, _ in val_dl:
        waveform = waveform.to(device)
        logits, features = model(waveform), model.hidden_features
        concat_features = torch.cat([logits, features], dim=1)
        exit()

