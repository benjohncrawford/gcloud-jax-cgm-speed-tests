#!/usr/bin/env bash
# Provision GCP VMs, run jax-gcm speed tests, and collect results.
#
# Usage:
#   chmod +x run_benchmarks.sh
#   ./run_benchmarks.sh
#
# Prerequisites:
#   - gcloud CLI authenticated with a project that has Compute Engine + TPU APIs enabled
#   - Sufficient quota for the requested resources in us-central1
#
# What this does:
#   1. Creates three VMs (TPU v5e, GPU P100, CPU c4-standard-32)
#   2. Copies speed_test.py and runs it on each
#   3. Downloads results JSON from each VM
#   4. Optionally deletes the VMs (prompted)

set -euo pipefail

PROJECT=$(gcloud config get-value project)
REGION=us-central1
ZONE_CPU=us-central1-a
ZONE_GPU=us-central1-c        # P100s are available in us-central1-c
ZONE_TPU=us-central1-a
RESULTS_DIR="./benchmark_results"

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo " jax-gcm Benchmark Suite"
echo " Project: $PROJECT"
echo "============================================"

# ---------- VM names ----------
VM_TPU="jcm-bench-tpu-v6e"
VM_GPU="jcm-bench-gpu-p100"
VM_CPU="jcm-bench-cpu-c4"

###############################################################################
# 1. Create VMs
###############################################################################

echo ""
echo "--- Creating VMs ---"

# TPU v5e (single-host, 1 chip)
# NOTE: If you get a permission error, you may need to request TPU quota first.
# Check available types:  gcloud compute tpus accelerator-types list --zone=$ZONE_TPU
# Request quota at:       https://console.cloud.google.com/iam-admin/quotas
# echo "Creating TPU v5e VM..."
# gcloud compute tpus tpu-vm create "$VM_TPU" \
#   --zone="$ZONE_TPU" \
#   --accelerator-type=v6e-1 \
#   --version=tpu-ubuntu2204-base \
#   --preemptible &

# # GPU P100 (Deep Learning VM with CUDA 12.8 + NVIDIA 570 drivers pre-installed)
# echo "Creating P100 GPU VM..."
# gcloud compute instances create "$VM_GPU" \
#   --zone="$ZONE_GPU" \
#   --machine-type=n1-standard-8 \
#   --accelerator=type=nvidia-tesla-p100,count=1 \
#   --maintenance-policy=TERMINATE \
#   --image-family=common-cu128-ubuntu-2204-nvidia-570 \
#   --image-project=deeplearning-platform-release \
#   --boot-disk-size=100GB \
#   --preemptible &

# CPU c4-standard-32
echo "Creating c4-standard-32 CPU VM..."
gcloud compute instances create "$VM_CPU" \
  --zone="$ZONE_CPU" \
  --machine-type=c4-standard-32 \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=50GB \
  --preemptible &

wait
echo "All VMs created."

###############################################################################
# 2. Setup scripts (per-device)
###############################################################################

read -r -d '' SETUP_CPU << 'SETUP_CPU_EOF' || true
#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-dev

if ! python3 -c "import sys; assert sys.version_info >= (3,11)" 2>/dev/null; then
    sudo apt-get install -y -qq software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null || true
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev
    PYTHON=python3.11
else
    PYTHON=python3
fi

$PYTHON -m venv ~/bench_env
source ~/bench_env/bin/activate
pip install --upgrade pip setuptools wheel
pip install jax jcm

cd ~
python speed_test.py --total_time 360 --save_interval 30 --n_repeats 5 \
  | tee benchmark_result.json
SETUP_CPU_EOF

read -r -d '' SETUP_GPU << 'SETUP_GPU_EOF' || true
#!/usr/bin/env bash
set -euo pipefail

# Deep Learning VM has CUDA 12.8 + NVIDIA 570 drivers pre-installed
nvidia-smi

sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-dev

if ! python3 -c "import sys; assert sys.version_info >= (3,11)" 2>/dev/null; then
    sudo apt-get install -y -qq software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null || true
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev
    PYTHON=python3.11
else
    PYTHON=python3
fi

$PYTHON -m venv ~/bench_env
source ~/bench_env/bin/activate
pip install --upgrade pip setuptools wheel
pip install "jax[cuda12]" jcm

cd ~
python speed_test.py --total_time 360 --save_interval 30 --n_repeats 5 \
  | tee benchmark_result.json
SETUP_GPU_EOF

read -r -d '' SETUP_TPU << 'SETUP_TPU_EOF' || true
#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-dev

if ! python3 -c "import sys; assert sys.version_info >= (3,11)" 2>/dev/null; then
    sudo apt-get install -y -qq software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null || true
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev
    PYTHON=python3.11
else
    PYTHON=python3
fi

$PYTHON -m venv ~/bench_env
source ~/bench_env/bin/activate
pip install --upgrade pip setuptools wheel
pip install "jax[tpu]" jcm

cd ~
python speed_test.py --total_time 360 --save_interval 30 --n_repeats 5 \
  | tee benchmark_result.json
SETUP_TPU_EOF

###############################################################################
# 3. Run benchmarks
###############################################################################

echo ""
echo "--- Running benchmarks ---"

run_benchmark() {
    local vm_name=$1
    local zone=$2
    local ssh_cmd=$3  # "gcloud compute ssh" or "gcloud compute tpus tpu-vm ssh"
    local setup_script=$4

    echo "[$vm_name] Waiting for SSH..."
    for i in $(seq 1 30); do
        if $ssh_cmd "$vm_name" --zone="$zone" --command="echo ready" 2>/dev/null; then
            break
        fi
        sleep 10
    done

    echo "[$vm_name] Copying speed_test.py..."
    if [[ "$ssh_cmd" == *"tpu-vm"* ]]; then
        gcloud compute tpus tpu-vm scp speed_test.py "$vm_name":~/speed_test.py --zone="$zone"
    else
        gcloud compute scp speed_test.py "$vm_name":~/speed_test.py --zone="$zone"
    fi

    echo "[$vm_name] Running setup + benchmark..."
    $ssh_cmd "$vm_name" --zone="$zone" --command="$setup_script"

    echo "[$vm_name] Downloading results..."
    if [[ "$ssh_cmd" == *"tpu-vm"* ]]; then
        gcloud compute tpus tpu-vm scp "$vm_name":~/benchmark_result.json \
            "$RESULTS_DIR/${vm_name}.json" --zone="$zone"
    else
        gcloud compute scp "$vm_name":~/benchmark_result.json \
            "$RESULTS_DIR/${vm_name}.json" --zone="$zone"
    fi

    echo "[$vm_name] Done."
}

# Run all three in parallel
# run_benchmark "$VM_TPU" "$ZONE_TPU" "gcloud compute tpus tpu-vm ssh" "$SETUP_TPU" &
# run_benchmark "$VM_GPU" "$ZONE_GPU" "gcloud compute ssh" "$SETUP_GPU" &
run_benchmark "$VM_CPU" "$ZONE_CPU" "gcloud compute ssh" "$SETUP_CPU" &
wait

echo ""
echo "============================================"
echo " All benchmarks complete!"
echo " Results saved to $RESULTS_DIR/"
echo "============================================"
ls -la "$RESULTS_DIR"

###############################################################################
# 4. Cleanup prompt
###############################################################################

echo ""
read -p "Delete all benchmark VMs? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Deleting VMs..."
    # gcloud compute tpus tpu-vm delete "$VM_TPU" --zone="$ZONE_TPU" --quiet &
    # gcloud compute instances delete "$VM_GPU" --zone="$ZONE_GPU" --quiet &
    gcloud compute instances delete "$VM_CPU" --zone="$ZONE_CPU" --quiet &
    wait
    echo "All VMs deleted."
else
    echo "VMs left running. Delete them manually when done:"
    echo "  gcloud compute tpus tpu-vm delete $VM_TPU --zone=$ZONE_TPU --quiet"
    echo "  gcloud compute instances delete $VM_GPU --zone=$ZONE_GPU --quiet"
    echo "  gcloud compute instances delete $VM_CPU --zone=$ZONE_CPU --quiet"
fi
