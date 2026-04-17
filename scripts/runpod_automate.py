#!/usr/bin/env python3
"""
RunPod Automation Script for Parameter Golf
============================================
Automates: Pod creation → Setup → SP8192 retokenization → All runs → Log download → Pod termination

Usage:
    export RUNPOD_API_KEY="your-key-here"
    python3 scripts/runpod_automate.py [--gpu 8xH100] [--tracks A,B,C] [--seeds 42,314,999] [--dry-run]

Budget: ~$50 covers all 3 tracks × 3 seeds on 8×H100 (~$21 each track, plus setup time)
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

RUNPOD_API = "https://api.runpod.io/graphql"
GITHUB_REPO = "https://github.com/tobiasoberrauch/parameter-golf.git"
BRANCH = "my-submission"

# GPU configurations and pricing (approx $/hr)
GPU_CONFIGS = {
    "1xH100": {"gpu_type_id": "NVIDIA H100 80GB HBM3", "gpu_count": 1, "price_hr": 3.0},
    "8xH100": {"gpu_type_id": "NVIDIA H100 80GB HBM3", "gpu_count": 8, "price_hr": 20.0},
    "1xA100": {"gpu_type_id": "NVIDIA A100 80GB PCIe", "gpu_count": 1, "price_hr": 1.5},
    "8xA100": {"gpu_type_id": "NVIDIA A100 80GB PCIe", "gpu_count": 8, "price_hr": 12.0},
}

TRACK_CONFIGS = {
    "A": {
        "name": "Track A: Incremental SOTA",
        "script": "records/track_10min_16mb/2026-04-17_TrackA_IncrementalSOTA/train_gpt.py",
        "env_extra": {},
        "est_minutes": 20,
    },
    "B": {
        "name": "Track B: Adaptive Recurrence",
        "script": "records/track_10min_16mb/2026-04-17_TrackB_AdaptiveRecurrence/train_gpt.py",
        "env_extra": {},
        "est_minutes": 22,
    },
    "C": {
        "name": "Track C: Mamba SSM",
        "script": "records/track_non_record_16mb/2026-04-17_TrackC_SubMamba50M/train_gpt.py",
        "env_extra": {
            "DATA_PATH": "./data/datasets/fineweb10B_sp8192",
            "TOKENIZER_PATH": "./data/tokenizers/fineweb_8192_bpe.model",
            "VOCAB_SIZE": "8192",
        },
        "est_minutes": 20,
    },
}


def runpod_query(api_key: str, query: str, variables: dict | None = None) -> dict:
    """Execute a RunPod GraphQL query."""
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{RUNPOD_API}?api_key={api_key}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode())
    if "errors" in result:
        raise RuntimeError(f"RunPod API error: {result['errors']}")
    return result["data"]


def create_pod(api_key: str, gpu_config: str, name: str = "param-golf") -> str:
    """Create a RunPod GPU pod and return its ID."""
    cfg = GPU_CONFIGS[gpu_config]
    query = """
    mutation {{
        podFindAndDeployOnDemand(input: {{
            name: "{name}",
            imageName: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
            gpuTypeId: "{gpu_type}",
            gpuCount: {gpu_count},
            volumeInGb: 100,
            containerDiskInGb: 50,
            minVcpuCount: 8,
            minMemoryInGb: 64,
            startSsh: true
        }}) {{
            id
            desiredStatus
            imageName
            machine {{
                podHostId
            }}
        }}
    }}
    """.format(name=name, gpu_type=cfg["gpu_type_id"], gpu_count=cfg["gpu_count"])
    result = runpod_query(api_key, query)
    pod = result["podFindAndDeployOnDemand"]
    return pod["id"]


def get_pod_status(api_key: str, pod_id: str) -> dict:
    """Get pod status and SSH connection info."""
    query = """
    query {{
        pod(input: {{ podId: "{pod_id}" }}) {{
            id
            desiredStatus
            runtime {{
                uptimeInSeconds
                ports {{
                    ip
                    isIpPublic
                    privatePort
                    publicPort
                    type
                }}
            }}
        }}
    }}
    """.format(pod_id=pod_id)
    result = runpod_query(api_key, query)
    return result["pod"]


def wait_for_pod_ready(api_key: str, pod_id: str, timeout: int = 600) -> dict:
    """Wait until pod is running and SSH is available."""
    print(f"  Warte auf Pod {pod_id}...", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        status = get_pod_status(api_key, pod_id)
        runtime = status.get("runtime")
        if runtime and runtime.get("ports"):
            ssh_ports = [p for p in runtime["ports"] if p["privatePort"] == 22]
            if ssh_ports:
                print(f"  Pod bereit! (nach {int(time.time()-start)}s)")
                return status
        time.sleep(10)
        print("  ...", end="", flush=True)
    raise TimeoutError(f"Pod {pod_id} nicht bereit nach {timeout}s")


def get_ssh_command(pod_status: dict) -> tuple[str, int]:
    """Extract SSH host and port from pod status."""
    for port in pod_status["runtime"]["ports"]:
        if port["privatePort"] == 22 and port["isIpPublic"]:
            return port["ip"], port["publicPort"]
    raise RuntimeError("Kein SSH-Port gefunden")


def ssh_exec(host: str, port: int, command: str, timeout: int = 3600) -> int:
    """Execute command on pod via SSH."""
    ssh_cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-p", str(port), f"root@{host}",
        command,
    ]
    print(f"  $ {command[:120]}{'...' if len(command) > 120 else ''}")
    result = subprocess.run(ssh_cmd, timeout=timeout)
    return result.returncode


def ssh_exec_stream(host: str, port: int, command: str, log_file: str | None = None, timeout: int = 3600) -> int:
    """Execute command on pod via SSH, streaming output."""
    ssh_cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-p", str(port), f"root@{host}",
        command,
    ]
    print(f"  $ {command[:120]}{'...' if len(command) > 120 else ''}")
    if log_file:
        with open(log_file, "w") as f:
            result = subprocess.run(ssh_cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    else:
        result = subprocess.run(ssh_cmd, timeout=timeout)
    return result.returncode


def scp_download(host: str, port: int, remote_path: str, local_path: str) -> int:
    """Download file from pod via SCP."""
    scp_cmd = [
        "scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-P", str(port), f"root@{host}:{remote_path}", local_path,
    ]
    print(f"  Download: {remote_path} → {local_path}")
    result = subprocess.run(scp_cmd, timeout=120)
    return result.returncode


def terminate_pod(api_key: str, pod_id: str):
    """Terminate (delete) a pod."""
    query = """
    mutation {{
        podTerminate(input: {{ podId: "{pod_id}" }})
    }}
    """.format(pod_id=pod_id)
    runpod_query(api_key, query)
    print(f"  Pod {pod_id} terminiert.")


def stop_pod(api_key: str, pod_id: str):
    """Stop a pod (keeps volume, no GPU charges)."""
    query = """
    mutation {{
        podStop(input: {{ podId: "{pod_id}" }})
    }}
    """.format(pod_id=pod_id)
    runpod_query(api_key, query)
    print(f"  Pod {pod_id} gestoppt.")


def estimate_cost(tracks: list[str], seeds: list[int], gpu_config: str) -> float:
    """Estimate total cost in USD."""
    cfg = GPU_CONFIGS[gpu_config]
    total_minutes = 0
    # Setup: ~15 min (clone, retokenize SP8192)
    total_minutes += 15
    for track in tracks:
        tc = TRACK_CONFIGS[track]
        total_minutes += tc["est_minutes"] * len(seeds)
    total_hours = total_minutes / 60
    return total_hours * cfg["price_hr"]


def run_setup(host: str, port: int):
    """Clone repo and prepare data on pod."""
    commands = [
        f"cd /workspace && git clone {GITHUB_REPO} && cd parameter-golf && git checkout {BRANCH}",
        "cd /workspace/parameter-golf && pip install sentencepiece 2>/dev/null",
        "cd /workspace/parameter-golf && python3 data/cached_challenge_fineweb.py --variant sp8192",
    ]
    for cmd in commands:
        rc = ssh_exec(host, port, cmd, timeout=1800)
        if rc != 0:
            raise RuntimeError(f"Setup-Befehl fehlgeschlagen (rc={rc}): {cmd}")


def run_track(host: str, port: int, track: str, seeds: list[int], gpu_count: int, local_log_dir: str):
    """Run a track with multiple seeds."""
    tc = TRACK_CONFIGS[track]
    print(f"\n{'='*60}")
    print(f"  {tc['name']}")
    print(f"{'='*60}")

    for seed in seeds:
        run_id = f"track_{track.lower()}_seed{seed}"
        log_name = f"train_{track.lower()}_seed{seed}.log"
        remote_log = f"/workspace/parameter-golf/{log_name}"

        env_parts = [f"SEED={seed}", f"RUN_ID={run_id}"]
        for k, v in tc["env_extra"].items():
            env_parts.append(f"{k}={v}")
        env_str = " ".join(env_parts)

        cmd = (
            f"cd /workspace/parameter-golf && "
            f"{env_str} torchrun --standalone --nproc_per_node={gpu_count} "
            f"{tc['script']} 2>&1 | tee {log_name}"
        )

        print(f"\n  Seed {seed}:")
        rc = ssh_exec(host, port, cmd, timeout=2400)
        if rc != 0:
            print(f"  WARNUNG: Run {run_id} exit code {rc}")

        # Download log
        local_log = os.path.join(local_log_dir, log_name)
        scp_download(host, port, remote_log, local_log)


def collect_results(host: str, port: int):
    """Print final results from pod."""
    cmd = "cd /workspace/parameter-golf && grep 'final_' train_*.log 2>/dev/null || echo 'Keine Ergebnisse gefunden'"
    ssh_exec(host, port, cmd)


def main():
    parser = argparse.ArgumentParser(description="RunPod Automation for Parameter Golf")
    parser.add_argument("--gpu", default="8xH100", choices=list(GPU_CONFIGS.keys()),
                        help="GPU configuration (default: 8xH100)")
    parser.add_argument("--tracks", default="A,B,C",
                        help="Comma-separated tracks to run (default: A,B,C)")
    parser.add_argument("--seeds", default="42,314,999",
                        help="Comma-separated seeds (default: 42,314,999)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show cost estimate without running")
    parser.add_argument("--keep-pod", action="store_true",
                        help="Don't terminate pod after runs")
    parser.add_argument("--pod-id", default=None,
                        help="Use existing pod instead of creating new one")
    parser.add_argument("--log-dir", default="./logs/runpod",
                        help="Local directory for downloaded logs")
    parser.add_argument("--skip-setup", action="store_true",
                        help="Skip clone/retokenize (pod already set up)")
    parser.add_argument("--tracks-only", default=None,
                        help="Run only specific tracks, e.g. 'A' for just Track A")
    args = parser.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: RUNPOD_API_KEY nicht gesetzt!")
        print("  export RUNPOD_API_KEY='your-key-here'")
        sys.exit(1)

    tracks = [t.strip().upper() for t in args.tracks.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    gpu_config = args.gpu
    gpu_count = GPU_CONFIGS[gpu_config]["gpu_count"]

    # Cost estimate
    est_cost = estimate_cost(tracks, seeds, gpu_config)
    print(f"\n{'='*60}")
    print(f"  Parameter Golf - RunPod Automation")
    print(f"{'='*60}")
    print(f"  GPU:      {gpu_config} (~${GPU_CONFIGS[gpu_config]['price_hr']:.0f}/h)")
    print(f"  Tracks:   {', '.join(tracks)}")
    print(f"  Seeds:    {', '.join(map(str, seeds))}")
    print(f"  Runs:     {len(tracks) * len(seeds)} total")
    print(f"  Kosten:   ~${est_cost:.0f} (geschätzt)")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n  [Dry Run - keine Aktion]")
        print(f"\n  Aufschlüsselung:")
        print(f"    Setup (clone + SP8192 retokenize):  ~15 min = ~${15/60 * GPU_CONFIGS[gpu_config]['price_hr']:.1f}")
        for track in tracks:
            tc = TRACK_CONFIGS[track]
            mins = tc["est_minutes"] * len(seeds)
            cost = mins / 60 * GPU_CONFIGS[gpu_config]["price_hr"]
            print(f"    {tc['name']} ({len(seeds)} seeds):  ~{mins} min = ~${cost:.1f}")
        return

    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)

    pod_id = args.pod_id
    created_pod = False

    try:
        # Step 1: Create or use existing pod
        if not pod_id:
            print("\n[1/5] Pod erstellen...")
            pod_id = create_pod(api_key, gpu_config, name=f"param-golf-{gpu_config}")
            created_pod = True
            print(f"  Pod ID: {pod_id}")
        else:
            print(f"\n[1/5] Nutze bestehenden Pod: {pod_id}")

        # Step 2: Wait for pod to be ready
        print("\n[2/5] Warte auf Pod...")
        status = wait_for_pod_ready(api_key, pod_id)
        host, port = get_ssh_command(status)
        print(f"  SSH: root@{host}:{port}")

        # Step 3: Setup
        if not args.skip_setup:
            print("\n[3/5] Setup (Clone + SP8192 Retokenization)...")
            run_setup(host, port)
        else:
            print("\n[3/5] Setup übersprungen (--skip-setup)")

        # Step 4: Run all tracks
        print("\n[4/5] Training Runs...")
        for track in tracks:
            run_track(host, port, track, seeds, gpu_count, args.log_dir)

        # Step 5: Collect results
        print("\n[5/5] Ergebnisse sammeln...")
        collect_results(host, port)

        print(f"\n  Logs gespeichert in: {args.log_dir}/")

    finally:
        # Terminate pod unless --keep-pod
        if pod_id and created_pod and not args.keep_pod:
            print("\n  Pod terminieren...")
            try:
                terminate_pod(api_key, pod_id)
            except Exception as e:
                print(f"  WARNUNG: Pod-Terminierung fehlgeschlagen: {e}")
                print(f"  Bitte manuell terminieren: Pod ID {pod_id}")
        elif pod_id and args.keep_pod:
            print(f"\n  Pod {pod_id} läuft weiter (--keep-pod)")

    print(f"\n{'='*60}")
    print(f"  FERTIG!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
