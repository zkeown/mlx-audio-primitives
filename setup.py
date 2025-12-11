"""
Setup script for mlx-audio-primitives with C++ extension.

Uses CMake to build the C++ extension module.
"""

from mlx import extension
from setuptools import setup

setup(
    # name and version are defined in pyproject.toml (single source of truth)
    ext_modules=[extension.CMakeExtension("mlx_audio_primitives._ext")],
    cmdclass={"build_ext": extension.CMakeBuild},
    packages=["mlx_audio_primitives"],
    package_data={"mlx_audio_primitives": ["*.so", "*.dylib", "*.metallib"]},
    zip_safe=False,
)
