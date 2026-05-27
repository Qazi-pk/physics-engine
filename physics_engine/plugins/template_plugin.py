"""
Example PIR Plugin Template

This file serves as a template for creating custom PIR Engine plugins.
Copy this file and modify it to create your own plugin.

Example usage:
    >>> from physics_engine.plugins import register_plugin
    >>> from my_plugin import MyCustomPlugin
    >>> 
    >>> plugin = MyCustomPlugin()
    >>> register_plugin(plugin)
"""

from __future__ import annotations

from typing import Any, Dict

from physics_engine.plugins.plugin_base import PIRPlugin


class ExamplePlugin(PIRPlugin):
    """
    Example plugin demonstrating how to extend PIR Engine.
    
    This plugin shows how to register custom models, datasets,
    pipelines, and operators.
    """
    
    # Plugin metadata
    name = "example_plugin"
    version = "0.1.0"
    description = "Example plugin showing how to extend PIR Engine"
    
    def register(self) -> None:
        """Register all plugin components."""
        self._models = self.register_models()
        self._datasets = self.register_datasets()
        self._pipelines = self.register_pipelines()
        self._operators = self.register_operators()
    
    def register_models(self) -> Dict[str, Any]:
        """
        Register custom physics models.
        
        Add your custom model classes here.
        """
        return {
            # Example: "custom_oscillator": CustomOscillatorModel,
        }
    
    def register_datasets(self) -> Dict[str, Any]:
        """
        Register custom dataset loaders.
        
        Add functions that load your custom datasets.
        """
        def load_example_data():
            """Load example dataset."""
            import pandas as pd
            return pd.DataFrame({
                'time': [0, 1, 2, 3, 4],
                'position': [0, 1, 4, 9, 16],
                'velocity': [0, 1, 2, 3, 4],
            })
        
        return {
            "example_data": load_example_data,
        }
    
    def register_pipelines(self) -> Dict[str, Any]:
        """
        Register custom discovery pipelines.
        
        Add your custom pipeline classes here.
        """
        return {
            # Example: "example_pipeline": ExampleDiscoveryPipeline,
        }
    
    def register_operators(self) -> Dict[str, Any]:
        """
        Register custom operators for symbolic discovery.
        
        Add custom mathematical operators.
        """
        return {
            "cube": {
                "func": lambda x: x ** 3,
                "arity": 1,
                "latex": r"x^3",
                "description": "Cube operator",
            },
            "sigmoid": {
                "func": lambda x: 1 / (1 + __import__('numpy').exp(-x)),
                "arity": 1,
                "latex": r"\sigma(x)",
                "description": "Sigmoid function",
            },
        }


# =============================================================================
# Advanced Example: Custom Physics Model
# =============================================================================

class CustomOscillatorModel:
    """
    Example custom physics model.
    
    This demonstrates how to create a model that can be used
    with PIR's discovery pipelines.
    """
    
    def __init__(self, omega: float = 1.0, damping: float = 0.1):
        """
        Initialize the oscillator model.
        
        Args:
            omega: Natural frequency
            damping: Damping coefficient
        """
        self.omega = omega
        self.damping = damping
    
    def predict(self, t, x0, v0):
        """
        Predict oscillator position.
        
        Args:
            t: Time array
            x0: Initial position
            v0: Initial velocity
        
        Returns:
            Position array
        """
        import numpy as np
        
        omega_d = self.omega * np.sqrt(1 - self.damping**2)
        A = x0
        B = (v0 + self.damping * self.omega * x0) / omega_d
        
        return np.exp(-self.damping * self.omega * t) * (
            A * np.cos(omega_d * t) + B * np.sin(omega_d * t)
        )
    
    def to_dict(self):
        """Export model parameters."""
        return {
            "type": "custom_oscillator",
            "omega": self.omega,
            "damping": self.damping,
        }


# =============================================================================
# Advanced Example: Custom Discovery Pipeline
# =============================================================================

class ExampleDiscoveryPipeline:
    """
    Example custom discovery pipeline.
    
    This demonstrates how to create a pipeline that integrates
    with PIR's discovery framework.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the pipeline.
        
        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
    
    def run(self, dataset):
        """
        Run the discovery pipeline.
        
        Args:
            dataset: Input dataset (pandas DataFrame or PIR Dataset)
        
        Returns:
            Discovered model or equation
        """
        if self.verbose:
            print("Running example discovery pipeline...")
        
        # Your discovery logic here
        # This is a placeholder that returns a simple result
        
        return {
            "equation": "y = a*x + b",
            "parameters": {"a": 1.0, "b": 0.0},
            "score": 0.95,
        }


# =============================================================================
# How to Use This Template
# =============================================================================

"""
To create your own plugin:

1. Copy this file to a new location:
   cp template_plugin.py my_awesome_plugin.py

2. Modify the ExamplePlugin class:
   - Change the name, version, and description
   - Implement register_models(), register_datasets(), etc.
   - Add your custom classes

3. Register your plugin:
   >>> from physics_engine.plugins import register_plugin
   >>> from my_awesome_plugin import MyAwesomePlugin
   >>> 
   >>> plugin = MyAwesomePlugin()
   >>> register_plugin(plugin)

4. Use your plugin components:
   >>> from physics_engine.plugins import get_plugin
   >>> 
   >>> plugin = get_plugin("my_awesome_plugin")
   >>> models = plugin.register_models()
   >>> my_model = models["my_custom_model"]()

5. Share your plugin:
   - Publish it as a separate package
   - Users can pip install your-plugin-package
   - Provide clear documentation and examples
"""


if __name__ == "__main__":
    # Test the plugin
    from physics_engine.plugins import register_plugin, list_plugins
    
    # Register the example plugin
    plugin = ExamplePlugin()
    register_plugin(plugin)
    
    # List all plugins
    plugins = list_plugins()
    print("Registered plugins:")
    for name, info in plugins.items():
        print(f"  - {name}: {info['description']}")
        print(f"    Version: {info['version']}")
        print(f"    Components: {len(info['models'])} models, "
              f"{len(info['datasets'])} datasets, "
              f"{len(info['pipelines'])} pipelines, "
              f"{len(info['operators'])} operators")
