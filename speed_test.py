"""Speed test for jax-gcm on different hardware architectures.

Usage:
    python speed_test.py [--total_time DAYS] [--save_interval DAYS] [--n_repeats N]

Runs a jcm simulation, pre-compiles, then times repeated runs.
Results are printed as JSON for easy aggregation.
"""

import argparse
import json
import platform
import time
import math

import jax
import jax.numpy as jnp
import numpy as np

from jcm.model import Model
from jcm.utils import VALID_TRUNCATIONS
from jcm.forcing import default_forcing
from jcm.terrain import TerrainData
from jcm.physics.speedy.speedy_coords import get_speedy_coords

NODAL_SHAPE_FOR_TRUNCATION = {
    21: (64, 32),
    31: (96, 48),
    42: (128, 64),
    85: (256, 128),
    106: (320, 160),
    119: (360, 180),
    170: (512, 256),
    213: (640, 320),
    340: (1024, 512),
    425: (1280, 640),
}

def block_until_ready(predictions):
    """Block until all arrays in a Predictions pytree are materialized."""
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        predictions,
    )
    return predictions


def get_device_info():
    """Gather JAX device and platform info."""
    devices = jax.devices()
    device = devices[0]
    return {
        "platform": jax.default_backend(),
        "device_kind": device.device_kind,
        "num_devices": len(devices),
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "hostname": platform.node(),
    }

def run_compile_time_test(total_time=360.0, n_repeats=10):
    compile_times = []
    run_times = []
    for i in range(n_repeats):
        jax.clear_caches()
        resolution = 21
        # --- Model setup ---
        print(f"Creating model (T{resolution})...")
        coords = get_speedy_coords(spectral_truncation=resolution)
        terrain = TerrainData.aquaplanet(coords=coords)
        model = Model(coords=coords, terrain=terrain)
        print("Model created.")

        print(f"Running model for {total_time} days...")
        t0 = time.perf_counter()
        _, predictions = model.run_from_state(initial_state=model._prepare_initial_modal_state(),
                                        forcing=default_forcing(model.coords.horizontal), 
                                        save_interval=total_time, 
                                        total_time=total_time)
        block_until_ready(predictions)
        compile_time = time.perf_counter() - t0
        compile_times.append(compile_time)
        print(f"Finished (includes compile): {compile_time:.2f}s")

    for i in range(n_repeats):
        print(f"Running model for {total_time} days...")
        t0 = time.perf_counter()
        _, predictions = model.run_from_state(initial_state=model._prepare_initial_modal_state(),
                                forcing=default_forcing(model.coords.horizontal), 
                                save_interval=total_time, 
                                total_time=total_time)
        block_until_ready(predictions)
        run_time = time.perf_counter() - t0
        run_times.append(run_time)
        print(f"Finished (does not include compile): {run_time:.2f}s")

    mean_runtime_with_compile = np.array(compile_times).mean()
    mean_runtime_no_compile = np.array(run_times).mean()

    print(f"Estimate compile time: {mean_runtime_with_compile-mean_runtime_no_compile}")
    return mean_runtime_with_compile-mean_runtime_no_compile

def run_speed_test(total_time=360.0, save_interval=30.0, n_repeats=5):
    """Run speed test and return results dict."""
    device_info = get_device_info()
    print(f"Device info: {json.dumps(device_info, indent=2)}")
    global_results = {**device_info,
                      "total_time_days": total_time,
                      "save_interval_days": save_interval,
                      "n_repeats": n_repeats
                     }
    for resolution in VALID_TRUNCATIONS:
        jax.clear_caches()
        # --- Model setup ---
        print(f"Creating model (T{resolution})...")
        coords = get_speedy_coords(nodal_shape=NODAL_SHAPE_FOR_TRUNCATION[resolution])
        terrain = TerrainData.aquaplanet(coords=coords)
        model = Model(coords=coords, terrain=terrain)
        print("Model created.")

        # --- Warmup / compile ---
        print(f"Warmup run ({total_time} days, save every {save_interval} days)...")
        t0 = time.perf_counter()
        _, predictions = model.run_from_state(initial_state=model._prepare_initial_modal_state(),
                                              forcing=default_forcing(model.coords.horizontal), 
                                              save_interval=save_interval, 
                                              total_time=total_time)
        block_until_ready(predictions)
        compile_time = time.perf_counter() - t0
        print(f"Output Size: {predictions.dynamics.u_wind.size}")
        print(f"Warmup (includes compile): {compile_time:.2f}s")

        # --- Timed runs ---
        print(f"Running {n_repeats} timed iterations...")
        times = []
        for i in range(n_repeats):
            t0 = time.perf_counter()
            _, predictions = model.run_from_state(initial_state=model._prepare_initial_modal_state(),
                                        forcing=default_forcing(model.coords.horizontal), 
                                        save_interval=save_interval, 
                                        total_time=total_time)
            block_until_ready(predictions)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            print(f"  Run {i + 1}/{n_repeats}: {elapsed:.3f}s")

        times = np.array(times)
        global_results[resolution] = {
            "resolution": resolution,
            "runtime_with_compilation_s": round(compile_time, 3),
            "mean_s": round(float(times.mean()), 3),
            "std_s": round(float(times.std()), 3),
            "min_s": round(float(times.min()), 3),
            "max_s": round(float(times.max()), 3),
            "all_times_s": [round(float(t), 3) for t in times],
        }

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(json.dumps(global_results, indent=2))
        with open("results.json", "w+") as f:
            f.write(json.dumps(global_results, indent=2))
        
    return global_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="jax-gcm speed test")
    parser.add_argument("--total_time", type=float, default=365.0, help="Simulation length in days")
    parser.add_argument("--save_interval", type=float, default=30.0, help="Save interval in days")
    parser.add_argument("--n_repeats", type=int, default=5, help="Number of timed repeats")
    parser.add_argument("--compile_test", type=bool, default=False, help="Run the compile time test (Does 10 iterations for total_time/10 days)")
    parser.add_argument("--run_time_test", type=bool, default=False, help="Run the run time test")
    args = parser.parse_args()

    if args.run_time_test:
        results = run_speed_test(
            total_time=args.total_time,
            save_interval=args.save_interval,
            n_repeats=args.n_repeats,
        )
    if args.compile_test:
        run_compile_time_test(math.floor(args.total_time/10), n_repeats=10)

