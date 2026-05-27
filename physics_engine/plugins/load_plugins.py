from __future__ import annotations

from physics_engine.plugins.plugin_registry import register_plugin
from physics_engine.plugins.robotics.robot_joint_plugin import RobotJointPlugin


def load_builtin_plugins() -> None:
    """
    Load built-in PIR plugins.

    This function registers the core set of plugins that ship with PIR.
    Additional domain-specific plugins can be registered separately.
    """
    register_plugin(RobotJointPlugin())
