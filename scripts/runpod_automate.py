#!/usr/bin/env python3
"""
RunPod Automation Script for Parameter Golf
============================================
Phased stealth strategy: Innovation → SOTA → Leaderboard

Phases:
    Phase 1 (Exploration):   Track B + A single seed → find what works      (~$25)
    Phase 2 (Optimization):  QK-Gain sweep on best track                    (~$35)
    Phase 3 (Validation):    3-seed official run with best config            (~$25)
    Phase 4 (Submission):    Submit PR only after results confirmed          (~$0)
    Reserve:                                                                 (~$15)

Usage:
    export RUNPOD_API_KEY="your-key-here"

    # Phase 1: Exploration (Innovation first)
    python3 scripts/runpod_automate.py --phase 1

    # Phase 2: Optimize best track from Phase 1
    python3 scripts/runpod_automate.py --phase 2 --best-track B

    # Phase 3: Validate with 3 seeds
    python3 scripts/runpod_automate.py --phase 3 --best-track B --best-qk 5.5

    # Or run everything manually
    python3 scripts/runpod_automate.py --tracks A,B --seeds 42 --dry-run

Budget: $100 total across all phases
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

# Phase definitions with budget tracking
PHASES = {
    1: {
        "name": "Exploration",
        "description": "Innovation first: Track B (novel) + Track A (baseline) — single seed each",
        "runs": [
            {"track": "B", "seeds": [42], "label": "Innovation (Adaptive SDClip + MoD)"},
            {"track": "A", "seeds": [42], "label": "Baseline SOTA comparison"},
        ],
        "budget": 25,
        "next": "Compare BPB: if B < A → optimize B. If A < B → optimize A. If close → optimize both.",
    },
    2: {
        "name": "Optimization",
        "description": "QK-Gain sweep (5.0–6.0) + TTT-LR sweep on best track from Phase 1",
        "runs": "dynamic",  # set at runtime based on --best-track
        "budget": 35,
        "next": "Pick best QK-Gain + TTT-LR combo. Proceed to Phase 3.",
    },
    3: {
        "name": "Validation",
        "description": "Official 3-seed run with optimized config",
        "runs": "dynamic",  # set at runtime based on --best-track + --best-qk
        "budget": 25,
        "next": "If BPB < 1.0810 → proceed to Phase 4 (submit PR). Otherwise reconsider.",
    },
    4: {
        "name": "Submission",
        "description": "Create PR with results. No GPU cost.",
        "runs": [],
        "budget": 0,
        "next": "Done. Watch leaderboard.",
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


def scp_download(host: str, port: int, remote_path: str, local_path: str) -> int:
    """Download file from pod via SCP."""
    scp_cmd = [
        "scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-P", str(port), f"root@{host}:{remote_path}", local_path,
    ]
    print(f"  Download: {remote_path} -> {local_path}")
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


def estimate_cost(runs: list[dict], gpu_config: str, include_setup: bool = True) -> float:
    """Estimate total cost in USD."""
    cfg = GPU_CONFIGS[gpu_config]
    total_minutes = 15 if include_setup else 0
    for run in runs:
        tc = TRACK_CONFIGS[run["track"]]
        total_minutes += tc["est_minutes"] * len(run["seeds"])
    return (total_minutes / 60) * cfg["price_hr"]


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


def run_single(host: str, port: int, track: str, seed: int, gpu_count: int,
               local_log_dir: str, env_overrides: dict | None = None, run_label: str = "") -> str:
    """Run a single training job. Returns local log path."""
    tc = TRACK_CONFIGS[track]
    run_id = f"track_{track.lower()}_seed{seed}"
    if run_label:
        run_id = run_label
    log_name = f"{run_id}.log"
    remote_log = f"/workspace/parameter-golf/{log_name}"

    env_parts = [f"SEED={seed}", f"RUN_ID={run_id}"]
    for k, v in tc["env_extra"].items():
        env_parts.append(f"{k}={v}")
    if env_overrides:
        for k, v in env_overrides.items():
            env_parts.append(f"{k}={v}")
    env_str = " ".join(env_parts)

    cmd = (
        f"cd /workspace/parameter-golf && "
        f"{env_str} torchrun --standalone --nproc_per_node={gpu_count} "
        f"{tc['script']} 2>&1 | tee {log_name}"
    )

    print(f"\n  [{run_id}] Seed={seed} Track={track}")
    if env_overrides:
        print(f"  Overrides: {env_overrides}")
    rc = ssh_exec(host, port, cmd, timeout=2400)
    if rc != 0:
        print(f"  WARNUNG: {run_id} exit code {rc}")

    local_log = os.path.join(local_log_dir, log_name)
    scp_download(host, port, remote_log, local_log)
    return local_log


def get_phase_runs(phase: int, args) -> list[dict]:
    """Get run configurations for a phase."""
    if phase == 1:
        return [
            {"track": "B", "seeds": [42], "env": {}, "label": "phase1_innovation_B"},
            {"track": "A", "seeds": [42], "env": {}, "label": "phase1_baseline_A"},
        ]
    elif phase == 2:
        track = args.best_track or "B"
        # Focused sweep: 3 QK values (skip known 5.25 SOTA, test above)
        qk_values = [5.25, 5.5, 5.75]
        runs = []
        for qk in qk_values:
            runs.append({
                "track": track,
                "seeds": [42],
                "env": {"QK_GAIN_INIT": str(qk)},
                "label": f"phase2_qk{qk}",
            })
        # One TTT-LR test (higher than default 0.005)
        runs.append({
            "track": track,
            "seeds": [42],
            "env": {"TTT_LR": "0.007"},
            "label": "phase2_tttlr0.007",
        })
        return runs
    elif phase == 3:
        track = args.best_track or "B"
        env = {}
        if args.best_qk:
            env["QK_GAIN_INIT"] = str(args.best_qk)
        if args.best_ttt_lr:
            env["TTT_LR"] = str(args.best_ttt_lr)
        return [
            {"track": track, "seeds": [42, 314, 999], "env": env, "label": f"phase3_final_{track}"},
        ]
    return []


def print_phase_plan(phase: int, runs: list[dict], gpu_config: str):
    """Print phase plan with cost estimate."""
    phase_info = PHASES[phase]
    est_runs = []
    for r in runs:
        seeds = r.get("seeds", [42])
        est_runs.append({"track": r["track"], "seeds": seeds})
    cost = estimate_cost(est_runs, gpu_config)

    print(f"\n{'='*60}")
    print(f"  Phase {phase}: {phase_info['name']}")
    print(f"  {phase_info['description']}")
    print(f"{'='*60}")
    print(f"  Budget: ~${phase_info['budget']} | Geschätzt: ~${cost:.0f}")
    print(f"  Runs:")
    for r in runs:
        seeds = r.get("seeds", [42])
        env = r.get("env", {})
        label = r.get("label", "")
        env_str = f" ({', '.join(f'{k}={v}' for k,v in env.items())})" if env else ""
        print(f"    - Track {r['track']}, Seeds {seeds}{env_str}  [{label}]")
    print(f"\n  Nächster Schritt: {phase_info['next']}")
    print(f"{'='*60}")


def print_budget_overview(completed_phases: list[int], current_phase: int, gpu_config: str):
    """Show budget overview across all phases."""
    print(f"\n  Budget-Übersicht ($100 total):")
    print(f"  {'Phase':<25} {'Budget':>8} {'Status':>12}")
    print(f"  {'-'*48}")
    total_spent = 0
    for p in [1, 2, 3, 4]:
        pi = PHASES[p]
        status = "done" if p in completed_phases else ("-> JETZT" if p == current_phase else "ausstehend")
        if p in completed_phases:
            total_spent += pi["budget"]
        print(f"  Phase {p}: {pi['name']:<17} ~${pi['budget']:>4}     {status}")
    print(f"  {'-'*48}")
    print(f"  {'Reserve':<25} ~${100 - sum(PHASES[p]['budget'] for p in [1,2,3,4]):>4}")
    print(f"  {'Ausgegeben':<25} ~${total_spent:>4}")
    print(f"  {'Verbleibend':<25} ~${100 - total_spent:>4}")


def collect_results(host: str, port: int):
    """Print final results from pod."""
    cmd = "cd /workspace/parameter-golf && grep 'final_' *.log 2>/dev/null || echo 'Keine Ergebnisse gefunden'"
    ssh_exec(host, port, cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Parameter Golf — Phased RunPod Automation (Stealth Strategy)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phasen:
  Phase 1: Exploration     Track B (Innovation) + A (Baseline), 1 Seed     ~$25
  Phase 2: Optimization    QK-Gain + TTT-LR Sweep auf bestem Track         ~$35
  Phase 3: Validation      3-Seed Official Run mit bestem Config            ~$25
  Phase 4: Submission      PR erstellen (kein GPU)                          ~$0

Beispiele:
  %(prog)s --phase 1                                    # Start: Innovation testen
  %(prog)s --phase 2 --best-track B                     # Optimiere Track B
  %(prog)s --phase 3 --best-track B --best-qk 5.75      # Finale 3-Seed Validierung
  %(prog)s --tracks A --seeds 42 --dry-run               # Manueller Modus
        """,
    )
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=None,
                        help="Run a specific phase (1=Explore, 2=Optimize, 3=Validate)")
    parser.add_argument("--best-track", default=None, choices=["A", "B"],
                        help="Best track from Phase 1 (for Phase 2+3)")
    parser.add_argument("--best-qk", type=float, default=None,
                        help="Best QK-Gain from Phase 2 sweep (for Phase 3)")
    parser.add_argument("--best-ttt-lr", type=float, default=None,
                        help="Best TTT-LR from Phase 2 sweep (for Phase 3)")
    parser.add_argument("--gpu", default="8xH100", choices=list(GPU_CONFIGS.keys()),
                        help="GPU configuration (default: 8xH100)")
    parser.add_argument("--tracks", default=None,
                        help="Manual mode: comma-separated tracks to run")
    parser.add_argument("--seeds", default="42,314,999",
                        help="Comma-separated seeds (default: 42,314,999)")
    parser.add_argument("--env", default=None, nargs="*",
                        help="Extra env vars, e.g. QK_GAIN_INIT=5.75 TTT_LR=0.007")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show cost estimate without running")
    parser.add_argument("--keep-pod", action="store_true",
                        help="Don't terminate pod after runs (saves setup for next phase)")
    parser.add_argument("--pod-id", default=None,
                        help="Use existing pod instead of creating new one")
    parser.add_argument("--log-dir", default="./logs/runpod",
                        help="Local directory for downloaded logs")
    parser.add_argument("--skip-setup", action="store_true",
                        help="Skip clone/retokenize (pod already set up)")
    args = parser.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: RUNPOD_API_KEY nicht gesetzt!")
        print("  export RUNPOD_API_KEY='your-key-here'")
        sys.exit(1)

    gpu_config = args.gpu
    gpu_count = GPU_CONFIGS[gpu_config]["gpu_count"]

    # Determine runs: phase mode or manual mode
    if args.phase:
        runs = get_phase_runs(args.phase, args)
        print_budget_overview(list(range(1, args.phase)), args.phase, gpu_config)
        print_phase_plan(args.phase, runs, gpu_config)
    elif args.tracks:
        tracks = [t.strip().upper() for t in args.tracks.split(",")]
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
        env_overrides = {}
        if args.env:
            for item in args.env:
                k, v = item.split("=", 1)
                env_overrides[k] = v
        runs = [{"track": t, "seeds": seeds, "env": env_overrides, "label": f"manual_{t}"} for t in tracks]

        # Cost display for manual mode
        est_runs = [{"track": r["track"], "seeds": r["seeds"]} for r in runs]
        cost = estimate_cost(est_runs, gpu_config)
        print(f"\n{'='*60}")
        print(f"  Parameter Golf - Manueller Modus")
        print(f"{'='*60}")
        print(f"  GPU:      {gpu_config} (~${GPU_CONFIGS[gpu_config]['price_hr']:.0f}/h)")
        print(f"  Tracks:   {', '.join(r['track'] for r in runs)}")
        print(f"  Seeds:    {seeds}")
        total_runs = sum(len(r["seeds"]) for r in runs)
        print(f"  Runs:     {total_runs} total")
        print(f"  Kosten:   ~${cost:.0f} (geschätzt)")
        if env_overrides:
            print(f"  Env:      {env_overrides}")
        print(f"{'='*60}")
    else:
        parser.print_help()
        print("\n  Tipp: Starte mit --phase 1 oder gib --tracks an.")
        return

    if args.dry_run:
        print("\n  [Dry Run — keine Aktion]")
        print(f"\n  Aufschlüsselung:")
        setup_cost = 15 / 60 * GPU_CONFIGS[gpu_config]["price_hr"]
        print(f"    Setup (clone + SP8192 retokenize):  ~15 min = ~${setup_cost:.1f}")
        for r in runs:
            tc = TRACK_CONFIGS[r["track"]]
            seeds = r.get("seeds", [42])
            mins = tc["est_minutes"] * len(seeds)
            cost = mins / 60 * GPU_CONFIGS[gpu_config]["price_hr"]
            label = r.get("label", r["track"])
            env = r.get("env", {})
            env_str = f" {env}" if env else ""
            print(f"    {label}{env_str}:  ~{mins} min = ~${cost:.1f}")
        return

    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)

    pod_id = args.pod_id
    created_pod = False

    try:
        # Step 1: Create or use existing pod
        if not pod_id:
            print("\n[1/4] Pod erstellen...")
            pod_id = create_pod(api_key, gpu_config, name=f"param-golf-{gpu_config}")
            created_pod = True
            print(f"  Pod ID: {pod_id}")
        else:
            print(f"\n[1/4] Nutze bestehenden Pod: {pod_id}")

        # Step 2: Wait for pod
        print("\n[2/4] Warte auf Pod...")
        status = wait_for_pod_ready(api_key, pod_id)
        host, port = get_ssh_command(status)
        print(f"  SSH: root@{host}:{port}")

        # Step 3: Setup
        if not args.skip_setup:
            print("\n[3/4] Setup (Clone + SP8192 Retokenization)...")
            run_setup(host, port)
        else:
            print("\n[3/4] Setup übersprungen (--skip-setup)")

        # Step 4: Run all jobs
        print("\n[4/4] Training Runs...")
        log_files = []
        for r in runs:
            track = r["track"]
            seeds = r.get("seeds", [42])
            env = r.get("env", {})
            label_base = r.get("label", f"track_{track.lower()}")
            for seed in seeds:
                label = f"{label_base}_s{seed}"
                log = run_single(host, port, track, seed, gpu_count,
                                 args.log_dir, env_overrides=env, run_label=label)
                log_files.append(log)

        # Results
        print(f"\n{'='*60}")
        print(f"  Ergebnisse:")
        print(f"{'='*60}")
        collect_results(host, port)
        print(f"\n  Logs: {args.log_dir}/")
        for lf in log_files:
            print(f"    {lf}")

        # Save pod ID for reuse
        if args.keep_pod:
            pod_file = os.path.join(args.log_dir, "pod_id.txt")
            with open(pod_file, "w") as f:
                f.write(pod_id)
            print(f"\n  Pod ID gespeichert: {pod_file}")
            print(f"  Nächste Phase: --pod-id {pod_id} --skip-setup")

    finally:
        if pod_id and created_pod and not args.keep_pod:
            print("\n  Pod terminieren...")
            try:
                terminate_pod(api_key, pod_id)
            except Exception as e:
                print(f"  WARNUNG: Pod-Terminierung fehlgeschlagen: {e}")
                print(f"  Manuell terminieren: Pod ID {pod_id}")
        elif pod_id and args.keep_pod:
            print(f"\n  Pod {pod_id} läuft weiter (--keep-pod)")

    # Phase-specific next steps
    if args.phase:
        phase_info = PHASES[args.phase]
        print(f"\n{'='*60}")
        print(f"  Phase {args.phase} abgeschlossen!")
        print(f"  Nächster Schritt: {phase_info['next']}")
        if args.phase == 1:
            print(f"\n  Vergleiche die Logs:")
            print(f"    grep 'final_' {args.log_dir}/phase1_*.log")
            print(f"\n  Dann:")
            print(f"    python3 scripts/runpod_automate.py --phase 2 --best-track <A|B> --keep-pod --pod-id {pod_id or '<POD_ID>'} --skip-setup")
        elif args.phase == 2:
            print(f"\n  Vergleiche QK-Gain Sweep:")
            print(f"    grep 'final_' {args.log_dir}/phase2_*.log")
            print(f"\n  Dann:")
            print(f"    python3 scripts/runpod_automate.py --phase 3 --best-track <A|B> --best-qk <WERT> --pod-id {pod_id or '<POD_ID>'} --skip-setup")
        elif args.phase == 3:
            print(f"\n  3-Seed Ergebnisse prüfen:")
            print(f"    grep 'final_' {args.log_dir}/phase3_*.log")
            print(f"\n  Wenn BPB < 1.0810: PR erstellen!")
        print(f"{'='*60}")
    else:
        print(f"\n  FERTIG!")


if __name__ == "__main__":
    main()
