from __future__ import annotations


class PIRPlugin:
    """
    Base interface for PIR plugins.

    Plugins extend PIR with domain-specific datasets, models, and pipelines
    without modifying the core engine.
    """

    name: str = "base_plugin"

    def register_datasets(self) -> dict[str, str]:
        """
        Return datasets provided by the plugin.

        Returns:
            dict mapping dataset name to description or path.
        """
        return {}

    def register_models(self) -> dict[str, str]:
        """
        Return physics models implemented by the plugin.

        Returns:
            dict mapping model name to description.
        """
        return {}

    def register_pipelines(self) -> dict[str, str]:
        """
        Return pipelines provided by the plugin.

        Returns:
            dict mapping pipeline name to module/class path.
        """
        return {}

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} '{self.name}'>"
