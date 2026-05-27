# subpackage
from . import mechanics
from .registry import registry
from .mechanics import NewtonSecondLaw, KinematicsVelocityLaw, KineticEnergyLaw

if not registry.list():
	registry.extend([
		NewtonSecondLaw(),
		KinematicsVelocityLaw(),
		KineticEnergyLaw(),
	])