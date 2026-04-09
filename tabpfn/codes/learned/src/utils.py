"""Checkpoint management utilities."""

import torch
from tfmplayground.model import NanoTabPFNModel


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device = None,
) -> tuple[NanoTabPFNModel, dict]:
    """Load a NanoTabPFNModel from a training checkpoint.

    Returns:
        (model, checkpoint_dict)
    """
    if device is None:
        device = torch.device("cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    arch = ckpt["architecture"]

    model = NanoTabPFNModel(
        embedding_size=arch["embedding_size"],
        num_attention_heads=arch["num_attention_heads"],
        mlp_hidden_size=arch["mlp_hidden_size"],
        num_layers=arch["num_layers"],
        num_outputs=arch["num_outputs"],
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt


def load_specialists(
    base_path: str,
    specialist_paths: list[str],
    device: torch.device = None,
) -> tuple[NanoTabPFNModel, list[NanoTabPFNModel], dict, list[dict]]:
    """Load base and specialist models.

    Returns:
        (base_model, specialist_models, base_state, specialist_states)
    """
    base_model, base_ckpt = load_model_from_checkpoint(base_path, device)
    base_state = base_ckpt["model"]

    specialist_models = []
    specialist_states = []
    for path in specialist_paths:
        model, ckpt = load_model_from_checkpoint(path, device)
        specialist_models.append(model)
        specialist_states.append(ckpt["model"])

    return base_model, specialist_models, base_state, specialist_states
