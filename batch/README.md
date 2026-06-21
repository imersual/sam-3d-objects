# Batch depth → SAM3D pipeline

Automates the full workflow for a folder of images: runs every depth backend
(Lotus-2, Depth Anything 3, Depth Pro, MoGe-2) in its conda env, converts each
to a SAM3D pointmap, then runs SAM3D once per backend (plus the MoGe v1 default).

## Folder layout

```
input/images/
  <name>/
    image.jpg        # the photo (jpg/jpeg/png)
    mask.png         # binary segmentation mask of the object (you provide it)
  <name2>/
    image.jpg
    mask.png

output/images/
  <name>/
    lotus2/    pointmap.pt, depth_metric.npy, ...
    da3/       pointmap.pt, depth.npy, intrinsics.npy, ...
    depthpro/  pointmap.pt, depth.npy, camera_params.json
    moge2/     pointmap.pt, depth.npy, mask.npy, camera_params.json
    splat_MoGe.glb               # SAM3D default (no pointmap; MoGe v1)
    splat_with_pt_lotus2.glb
    splat_with_pt_da3.glb
    splat_with_pt_depthpro.glb
    splat_with_pt_moge2.glb
```

## Prerequisites (on the remote box)

The conda/mamba envs from the setup docs must exist, and the depth-model
repos must be cloned. MoGe-2 needs no separate env — the `MoGe` package is
already a dependency of `sam3d-objects` (SAM3D's default uses MoGe v1):

| env             | repo                          | provides                  |
| --------------- | ----------------------------- | ------------------------- |
| `sam3d-objects` | this repo                     | SAM3D + MoGe-2 + pointmap |
| `lotus2`        | `/workspace/Lotus-2`          | `infer.py`                |
| `da3`           | `/workspace/Depth-Anything-3` | `depth_anything_3`        |
| `depthpro`      | `/workspace/ml-depth-pro`     | `depth_pro`               |

Paths and env names are configurable in [`config.sh`](config.sh).

## Run

```bash
cd /workspace/sam-3d-objects/batch

# process every folder under input/images/
./run_batch.sh

# process only specific samples
./run_batch.sh chair lamp

# override config inline
RUN_DEPTHPRO=0 GPU=1 LOTUS_FOV_H=70 LOTUS_NEAR=0.5 LOTUS_FAR=5.0 ./run_batch.sh
```

Each stage runs in a subshell with the right env activated; a failure in one
stage or one image is logged and skipped — the batch keeps going.

## Tuning knobs (see `config.sh`)

- `LOTUS_FOV_H` / `LOTUS_NEAR` / `LOTUS_FAR` — Lotus-2 has no scale or
  intrinsics, so these define the metric depth range. Defaults target
  product/food shots (`24°`, `0.2–1.5 m`). Use `70°`, `0.5–5.0 m` for rooms.
- `DA3_CONF_PERCENTILE` — discard the bottom N% confidence pixels from DA3.
- `MOGE2_MODEL` — MoGe-2 HF model id (default `Ruicheng/moge-2-vitl-normal`).
- `RUN_LOTUS` / `RUN_DA3` / `RUN_DEPTHPRO` / `RUN_MOGE2` / `RUN_SAM3D` — toggle stages (`0`/`1`).
- `GPU` — which GPU (`CUDA_VISIBLE_DEVICES`).
- `SAM3D_SEED` — reconstruction seed.

## Running stages standalone

Each script is independently runnable in its env, e.g.:

```bash
conda activate da3
python run_da3.py --image input/images/chair/image.jpg --out-dir output/images/chair/da3

conda activate sam3d-objects
python run_moge2.py --image input/images/chair/image.jpg --out-dir output/images/chair/moge2

conda activate sam3d-objects
python batch_sam3d.py --image input/images/chair/image.jpg \
    --mask input/images/chair/mask.png --out-dir output/images/chair
```
