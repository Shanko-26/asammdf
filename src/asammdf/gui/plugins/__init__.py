"""Plugin system for asammdf GUI"""

from .base import BasePlugin
from .manager import PluginManager

__all__ = ["BasePlugin", "PluginManager"]