from functools import lru_cache
import os
from pathlib import Path

@lru_cache(maxsize=1)
def get_package_root() -> Path:
    """Get the absolute path to the package root directory (cached)."""
    # Get the directory where this current module is located
    package_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # Look for a project root marker (like pyproject.toml or setup.py)
    # Adjust these markers based on your project structure
    root_markers = [
        "pyproject.toml",
        "setup.py",
        "requirements.txt",
        ".git",
        "models"  # If you have a models directory at root
    ]
    
    # Traverse up the directory tree until we find a root marker
    current = package_dir
    while current != current.parent:  # Stop at filesystem root
        if any((current / marker).exists() for marker in root_markers):
            return current
        current = current.parent
    
    # If no marker found, return the directory containing this module
    return package_dir

def resource_path(relative_path: str) -> Path:
    """Return absolute path to a resource file."""
    # First try relative to package root
    abs_path = get_package_root() / relative_path
    if abs_path.exists():
        return abs_path
    
    # Fallback: try relative to current working directory
    cwd_path = Path.cwd() / relative_path
    if cwd_path.exists():
        return cwd_path
    
    # Final fallback: try relative to this module's directory
    module_path = Path(__file__).parent / relative_path
    return module_path
