#!/bin/bash

# Usage: ./scripts/run_amd_dev.sh <image_name> <device_id> [command]

IMAGE_NAME="$1"
DEVICE_ID="$2"

if [ -z "$IMAGE_NAME" ]; then
    echo "Error: You must provide an image name."
    exit 1
fi

# Safeguard: Check if DEVICE_ID is a number or the exact string "all"
if ! [[ "$DEVICE_ID" =~ ^[0-9]+$ ]] && [[ "$DEVICE_ID" != "all" ]]; then
    echo "Error: Invalid device ID ('$DEVICE_ID')."
    echo "Usage: ./scripts/run_amd_dev.sh <image_name> <device_id> [command]"
    echo "Example: ./scripts/run_amd_dev.sh my_image all bash"
    exit 1
fi

# Remove the image name from the arguments list so we can pass the rest
shift 2

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# --- HARDWARE & DRIVER ACCESS ---
# --device=/dev/kfd:     (CRITICAL) Passes the AMD Compute module into the container.
# --device=/dev/dri:     (CRITICAL) Passes the Direct Rendering Infrastructure (the physical GPU).
# --group-add=video:     Grants basic display output access.
# --group-add=render:    (CRITICAL ON UBUNTU 24.04) Grants access to the DRM compute nodes.

# --- KERNEL & STABILITY FLAGS ---
# --ipc=host:            Prevents PyTorch DataLoader "Bus Error" crashes by allowing unlimited
#                        shared memory for multi-processing (num_workers > 0).
# --cap-add=SYS_PTRACE:  Allows ROCm profiling tools (like rocm-smi) to inspect GPU processes.
# --security-opt seccomp=unconfined: Prevents Docker's default security profile from blocking
#                                    low-level ROCm system calls during matrix math.
#
# --- MEMORY ALLOCATION & EXPERIMENTAL KERNELS ---
# PYTORCH_ALLOC_CONF: Prevents VRAM fragmentation OOMs. 'expandable_segments' allows
#                         PyTorch to dynamically resize cached memory blocks, and
#                         'garbage_collection_threshold' forces aggressive cleanups before failing.
# TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL: Forces ROCm to use highly optimized datacenter
#                                          Triton Flash Attention kernels on consumer RDNA3 cards.
#                                          (Disable immediately if training becomes unstable).
# --- AMD ROCm ENVIRONMENT VARIABLES ---
# HIP_VISIBLE_DEVICES:   Locks the runtime to specific cards. Set to '0' to explicitly bind to
#                        the primary 7900XT and ignore integrated CPU graphics.
# HSA_OVERRIDE_GFX_VERSION: (CRITICAL FOR 7900XT) ROCm officially targets Instinct datacenter cards.
#                           Setting this to '11.0.0' forces PyTorch to recognize the RDNA3 7900XT
#                           as a fully supported gfx1100 architecture.

# --- MOUNTS & NETWORKING ---
# --network=host:        Bypasses Docker's network bridge for zero-latency WandB telemetry.
# -v ~/.cache/huggingface:/workspace/cache: Aligns with Dockerfile HF_HOME to prevent re-downloads.

# --- OTHER -----
# -e DDP_SHARED_ID:         a common ID for all ranks in DDP, used to coordinate logging, etc. Can't use e.g.
#                           os.getppid or such because of Docker deterministic PID namespaces. Can't use
#                           TORCHELASTIC_RUN_ID without manually also setting --rdzv-id. Easiest just to pass directly.

docker run -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --group-add=render \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --network=host \
  -e DDP_SHARED_ID=$RANDOM \
  -e HIP_VISIBLE_DEVICES=$DEVICE_ID \
  -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  -e PYTORCH_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8" \
  -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  -v "$PROJECT_ROOT":/workspace \
  -v ~/.cache/huggingface:/workspace/cache \
  "$IMAGE_NAME" \
  "$@"
