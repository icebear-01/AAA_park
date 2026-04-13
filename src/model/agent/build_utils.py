from copy import deepcopy

import torch

from configs import ACTOR_CONFIGS, CRITIC_CONFIGS, UNPARK_IMG_MODE


def clone_network_configs():
    return deepcopy(ACTOR_CONFIGS), deepcopy(CRITIC_CONFIGS)


def set_img_channels(actor_layers, critic_layers, img_channels: int):
    for layers in (actor_layers, critic_layers):
        img_shape = layers.get("img_shape")
        if img_shape is None:
            continue
        _, width, height = img_shape
        layers["img_shape"] = (img_channels, width, height)
    return actor_layers, critic_layers


def configure_unpark_img_mode(actor_layers, critic_layers, img_mode: str):
    img_mode = normalize_unpark_img_mode(img_mode)
    img_channels = 4 if img_mode == "rgb_slot" else 3
    return set_img_channels(actor_layers, critic_layers, img_channels)


def normalize_unpark_img_mode(img_mode: str = None, use_slot_channel: bool = False):
    if img_mode is None:
        img_mode = UNPARK_IMG_MODE
    img_mode = str(img_mode).strip().lower()
    if use_slot_channel and img_mode == "rgb":
        img_mode = "rgb_slot"
    valid_modes = {"rgb", "rgb_slot", "occ_grid"}
    if img_mode not in valid_modes:
        raise ValueError("Unsupported unparking image mode: %s" % img_mode)
    return img_mode


def load_checkpoint_metadata(ckpt_path: str):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return {}

    configs = checkpoint.get("configs")
    if configs is None:
        return {}

    return {
        "configs": configs,
        "observation_shape": deepcopy(getattr(configs, "observation_shape", None)),
        "action_dim": getattr(configs, "action_dim", None),
        "actor_layers": deepcopy(getattr(configs, "actor_layers", None)),
        "critic_layers": deepcopy(getattr(configs, "critic_layers", None)),
        "unpark_img_mode": getattr(configs, "unpark_img_mode", None),
    }


def infer_slot_channel_from_ckpt(ckpt_path: str):
    return infer_unpark_img_mode_from_ckpt(ckpt_path) == "rgb_slot"


def infer_unpark_img_mode_from_ckpt(ckpt_path: str):
    metadata = load_checkpoint_metadata(ckpt_path)
    if metadata.get("unpark_img_mode") is not None:
        return normalize_unpark_img_mode(metadata["unpark_img_mode"])

    actor_layers = metadata.get("actor_layers")
    if actor_layers is None:
        return None

    img_shape = actor_layers.get("img_shape")
    if img_shape is None:
        return "rgb"
    if int(img_shape[0]) > 3:
        return "rgb_slot"
    return "rgb"


def resolve_agent_init_configs(
    observation_shape,
    action_dim,
    ckpt_path: str = None,
    actor_layers=None,
    critic_layers=None,
    extra_configs=None,
):
    resolved_actor_layers = deepcopy(actor_layers) if actor_layers is not None else deepcopy(ACTOR_CONFIGS)
    resolved_critic_layers = deepcopy(critic_layers) if critic_layers is not None else deepcopy(CRITIC_CONFIGS)
    resolved_observation_shape = deepcopy(observation_shape)
    resolved_action_dim = action_dim
    resolved_extra_configs = {} if extra_configs is None else deepcopy(extra_configs)

    if ckpt_path is not None:
        metadata = load_checkpoint_metadata(ckpt_path)
        if metadata.get("actor_layers") is not None:
            resolved_actor_layers = metadata["actor_layers"]
        if metadata.get("critic_layers") is not None:
            resolved_critic_layers = metadata["critic_layers"]
        if metadata.get("observation_shape") is not None:
            resolved_observation_shape = metadata["observation_shape"]
        if metadata.get("action_dim") is not None:
            resolved_action_dim = metadata["action_dim"]
        if metadata.get("unpark_img_mode") is not None:
            resolved_extra_configs["unpark_img_mode"] = normalize_unpark_img_mode(metadata["unpark_img_mode"])

    configs = {
        "discrete": False,
        "observation_shape": resolved_observation_shape,
        "action_dim": resolved_action_dim,
        "hidden_size": 64,
        "activation": "tanh",
        "dist_type": "gaussian",
        "save_params": False,
        "actor_layers": resolved_actor_layers,
        "critic_layers": resolved_critic_layers,
    }
    configs.update(resolved_extra_configs)
    return configs
