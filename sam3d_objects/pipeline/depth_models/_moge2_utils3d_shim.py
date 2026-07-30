# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Compatibility shim: expose ``utils3d.pt`` on top of SAM3D's pinned utils3d.

MoGe-2 (``moge.model.v2``) reads two functions from the ``utils3d.pt`` namespace
during inference:

    utils3d.pt.intrinsics_from_focal_center(...)   # moge/model/v2.py:266
    utils3d.pt.depth_map_to_point_map(...)          # moge/model/v2.py:276

That ``.pt`` namespace only exists in newer utils3d releases whose reorganized
API drops the ``utils3d.torch`` / ``utils3d.numpy`` / ``utils3d.io`` functions
SAM3D's rendering and mesh-postprocessing code depends on (``rasterize_triangle_faces``,
``depth_edge``, ``write_ply``, the mesh-topology helpers, ...). Upgrading utils3d
to get ``.pt`` would therefore break SAM3D.

SAM3D's pinned utils3d already ships both functions MoGe-2 needs -- one under the
same name, the other under its older name ``depth_to_points``. So rather than move
utils3d, we synthesize the tiny ``utils3d.pt`` surface MoGe-2 uses from what the
pinned utils3d already provides. SAM3D's own utils3d call sites are untouched.

``install()`` is idempotent and a no-op if a real ``utils3d.pt`` already exists
(e.g. if utils3d is later upgraded to a version that has it natively).
"""
import sys
import types

import utils3d
import utils3d.torch  # noqa: F401  ensure the backend submodule is imported


def _existing_pt():
    """Return a real ``utils3d.pt`` if this utils3d has one, else None.

    ``getattr(utils3d, "pt", None)`` is NOT safe here: utils3d's ``__init__``
    defines a lazy ``__getattr__`` that calls ``importlib.import_module('.pt')``,
    and getattr's default only suppresses AttributeError -- the ModuleNotFoundError
    raised for a missing submodule propagates instead.
    """
    try:
        return getattr(utils3d, "pt")
    except (AttributeError, ImportError):
        return None


def install():
    """Ensure ``utils3d.pt`` exposes the functions MoGe-2 calls. Idempotent."""
    if _existing_pt() is not None:
        return

    pt = types.ModuleType("utils3d.pt")
    # Same name across utils3d versions.
    pt.intrinsics_from_focal_center = utils3d.torch.intrinsics_from_focal_center
    # Renamed across utils3d versions: newer calls it depth_map_to_point_map,
    # SAM3D's pinned utils3d calls it depth_to_points. Prefer the new name if the
    # installed utils3d happens to provide it; otherwise fall back to the old one.
    pt.depth_map_to_point_map = getattr(
        utils3d.torch, "depth_map_to_point_map", None
    ) or utils3d.torch.depth_to_points

    utils3d.pt = pt
    sys.modules["utils3d.pt"] = pt
