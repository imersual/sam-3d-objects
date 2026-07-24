# Copyright (c) Meta Platforms, Inc. and affiliates.
from .base import DepthModel
from . import _moge2_utils3d_shim


class MoGe2(DepthModel):
    """MoGe-2 depth model.

    Unlike the v1 wrapper, no inference kwargs are passed: MoGe-2's defaults are
    already the best-quality settings (``resolution_level=9`` maps to the top of
    the token range, and ``apply_mask=True`` marks invalid pixels ``inf``, which
    is SAM3D's own convention -- see ``pipeline/utils/pointmap.py``). This
    matches how ``batch/run_moge2.py`` calls the model.

    MoGe-2 reads ``utils3d.pt.*`` during inference, a namespace SAM3D's pinned
    utils3d predates; ``_moge2_utils3d_shim.install()`` synthesizes it from the
    functions that utils3d already ships, so SAM3D stays on its own utils3d
    unchanged. Installed at construction, before any ``infer()`` call.
    """

    def __init__(self, model, device="cuda"):
        _moge2_utils3d_shim.install()
        super().__init__(model, device)

    def __call__(self, image):
        output = self.model.infer(image.to(self.device))
        output["pointmaps"] = output["points"]
        return output
