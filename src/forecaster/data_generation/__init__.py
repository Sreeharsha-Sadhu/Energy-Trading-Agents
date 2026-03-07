# src/data_generation/__init__.py
"""
Energy Load Data Generation package.
Modules:
 - load_factors: base load generation factors
 - solar_model: solar irradiance model
 - load_factors: hourly / seasonal factors
 - tracking: load/save tracking state
 - writer: chunked csv writer
 - energy_load_generator: orchestrator class
"""
__all__ = [
    "get_seasonal_factor",    "solar_model",
    "load_factors",
    "tracking",
    "writer",
    "energy_load_generator",
]
