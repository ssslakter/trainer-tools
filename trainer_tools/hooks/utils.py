from omegaconf import OmegaConf


def remove_disabled_hooks(config):
    """Remove disabled hooks from config to avoid interpolation errors.

    Creates a copy of the config and removes any hooks where enabled=False.
    This prevents OmegaConf from trying to resolve interpolations in disabled hooks
    that might reference non-existent config keys.

    Args:
        config: OmegaConf config object

    Returns:
        OmegaConf config with disabled hooks removed
    """
    # Create a copy without resolving interpolations
    config_copy = OmegaConf.create(OmegaConf.to_container(config, resolve=False))

    if "hooks" in config_copy:
        hooks_to_remove = []
        for hook_name in config_copy.hooks:
            hook = config_copy.hooks[hook_name]
            if hook is not None and not hook.get("enabled", False):
                hooks_to_remove.append(hook_name)
        for hook_name in hooks_to_remove:
            config_copy.hooks.pop(hook_name)

    return config_copy
