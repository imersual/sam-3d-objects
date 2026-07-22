# Copyright (c) Meta Platforms, Inc. and affiliates.
from .base import DepthModel


class MoGe2(DepthModel):
    """MoGe-2 depth model.

    Unlike the v1 wrapper, no inference kwargs are passed: MoGe-2's defaults are
    already the best-quality settings (``resolution_level=9`` maps to the top of
    the token range, and ``apply_mask=True`` marks invalid pixels ``inf``, which
    is SAM3D's own convention -- see ``pipeline/utils/pointmap.py``). This
    matches how ``batch/run_moge2.py`` calls the model.
    """

    def __call__(self, image):
        output = self.model.infer(image.to(self.device))
        output["pointmaps"] = output["points"]
        return output
