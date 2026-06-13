# core.identity package
import sys
import os

# Allow importing from project root (for siamese_vit.py which lives at top level)
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from siamese_vit import ViTBackbone as ViTFaceEmbedder
from .database import load_db, add_identity, add_identity_with_embedding

__all__ = [
    "ViTFaceEmbedder",
    "load_db",
    "add_identity",
    "add_identity_with_embedding",
]
