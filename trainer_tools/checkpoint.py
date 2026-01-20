from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from .imports import *
import logging

log = logging.getLogger(__name__)

def build_model_from_config(config: DictConfig) -> nn.Module:
    return instantiate(config)


def save_pretrained(model: nn.Module, save_dir: str, config: Optional[DictConfig] = None):
    """
    Save model and config to a directory
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model_path = save_path / "model.pt"
    torch.save({"model": model.state_dict()}, model_path)
    log.info(f"Saved model to {model_path}")

    if config is None: return
    config_path = save_path / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(config, f, resolve=True)
    log.info(f"Saved config to {config_path}")


def load_from_pretrained(
    model_dir: str, device: Optional[torch.device] = None, weights_only: bool = True, return_model: bool = True
):
    """
    Load model from a directory

    Args:
        model_dir: Directory containing model.pt and config.yaml
        device: Device to map tensors to (default: CPU)
        weights_only: Whether to use weights_only mode for torch.load
        return_model: If True, returns instantiated model. If False, returns (state_dict, config)

    Returns:
        If return_model=True: nn.Module (loaded model ready to use)
        If return_model=False: tuple (state_dict, config)
    """
    model_path = Path(model_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Load model weights - try both naming conventions
    checkpoint_path = model_path / "model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = model_path / "model_final.pt"
    if not checkpoint_path.exists():
        # Find any .pt file
        pt_files = list(model_path.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt file found in {model_dir}")
        checkpoint_path = pt_files[0]

    log.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu", weights_only=weights_only)

    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    config_path = model_path / "config.yaml"
    config = None
    if config_path.exists():
        log.info(f"Loading config from {config_path}")
        config = OmegaConf.load(config_path)
    else:
        log.warning(f"No config.yaml found in {model_dir}")

    if not return_model:
        return state_dict, config

    if config is None:
        raise ValueError(
            f"Cannot build model without config. No config.yaml found in {model_dir}. "
            "Use return_model=False to get state_dict only."
        )

    model = build_model_from_config(config)
    model.load_state_dict(state_dict)

    if device is not None:
        model = model.to(device)

    model.eval()
    log.info(f"Model loaded and ready on {device or 'cpu'}")

    return model
