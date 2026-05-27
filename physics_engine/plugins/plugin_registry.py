from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from physics_engine.plugins.plugin_base import PIRPlugin

PLUGIN_REGISTRY: list[PIRPlugin] = []


def register_plugin(plugin: PIRPlugin) -> None:
    """
    Register a plugin with the PIR engine.

    Args:
        plugin: A PIRPlugin instance to register.
    """
    if not hasattr(plugin, "name"):
        raise ValueError(f"Plugin {plugin} must have a 'name' attribute.")
    PLUGIN_REGISTRY.append(plugin)


def get_plugins() -> list[PIRPlugin]:
    """
    Get all registered plugins.

    Returns:
        List of all registered PIRPlugin instances.
    """
    return list(PLUGIN_REGISTRY)


def get_plugin_by_name(name: str) -> PIRPlugin | None:
    """
    Get a plugin by its name.

    Args:
        name: The plugin name.

    Returns:
        The PIRPlugin instance if found, else None.
    """
    for plugin in PLUGIN_REGISTRY:
        if plugin.name == name:
            return plugin
    return None


def list_plugin_capabilities() -> dict[str, dict]:
    """
    Get all capabilities registered by plugins.

    Returns:
        Dictionary with plugin names and their registered datasets, models, pipelines.
    """
    capabilities = {}
    for plugin in PLUGIN_REGISTRY:
        capabilities[plugin.name] = {
            "datasets": plugin.register_datasets(),
            "models": plugin.register_models(),
            "pipelines": plugin.register_pipelines(),
        }
    return capabilities
