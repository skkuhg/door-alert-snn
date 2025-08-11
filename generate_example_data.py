import argparse
import csv
import json
from pathlib import Path

import numpy as np


def generate_door_stream(T, dt, scenario='mixed'):
    """
    Generate synthetic door open/close events.

    Returns
    -------
    door: np.ndarray[int]
        Binary array, 1=open, 0=closed
    segments: list[tuple[int, int]]
        List of (start_idx, end_idx) open segments
    """
    door = np.zeros(T, dtype=np.int32)
    segments = []

    if scenario == 'mixed':
        # Scenario 1: Short open (5 seconds) - should NOT trigger alert
        start1 = int(2 * 60 / dt)  # Start at 2 minutes
        end1 = start1 + int(5 / dt)
        door[start1:end1] = 1
        segments.append((start1, end1))

        # Scenario 2: Medium open (15 seconds) - should NOT trigger alert
        start2 = int(5 * 60 / dt)  # Start at 5 minutes
        end2 = start2 + int(15 / dt)
        door[start2:end2] = 1
        segments.append((start2, end2))

        # Scenario 3: Long open (30 seconds) - SHOULD trigger alert at 20s
        start3 = int(8 * 60 / dt)  # Start at 8 minutes
        end3 = start3 + int(30 / dt)
        door[start3:end3] = 1
        segments.append((start3, end3))

        # Scenario 4: Very long open (45 seconds) - SHOULD trigger alert
        start4 = int(12 * 60 / dt)  # Start at 12 minutes
        end4 = start4 + int(45 / dt)
        door[start4:end4] = 1
        segments.append((start4, end4))

        # Scenario 5: Rapid open-close-open pattern
        start5 = int(16 * 60 / dt)  # Start at 16 minutes
        for i in range(3):
            s = start5 + i * int(8 / dt)
            e = s + int(6 / dt)
            door[s:e] = 1
            segments.append((s, e))

        # Scenario 6: Another long open (25 seconds) near the end
        start6 = int(20 * 60 / dt)  # Start at 20 minutes
        end6 = start6 + int(25 / dt)
        door[start6:end6] = 1
        segments.append((start6, end6))

    elif scenario == 'stochastic':
        # Stochastic door behavior with occasional long opens
        state = 0  # Start closed
        i = 0
        while i < T:
            if state == 0:  # Currently closed
                # Probability of opening
                if np.random.random() < 0.002:
                    state = 1
                    start = i
            else:  # Currently open
                # Determine open duration
                if np.random.random() < 0.01:  # Small chance of long open
                    duration = np.random.randint(25, 50) / dt
                else:
                    duration = np.random.randint(3, 18) / dt

                end = min(i + int(duration), T)
                door[i:end] = 1
                segments.append((i, end))
                i = end
                state = 0
            i += 1
    else:
        raise ValueError("scenario must be 'mixed' or 'stochastic'")

    return door, segments


def simulate_lif_timer(door, dt, tau_m, v_th, v_reset, bias_current, cooldown_s):
    """
    Simulate a single LIF neuron as a timer for door-open detection.

    Returns
    -------
    V: np.ndarray[float]
    alert: np.ndarray[int]
    cooldown_trace: np.ndarray[int]
    """
    T = len(door)
    V = np.zeros(T, dtype=np.float32)
    alert = np.zeros(T, dtype=np.int32)
    cooldown_trace = np.zeros(T, dtype=np.int32)

    cooldown_until = -np.inf

    for t in range(1, T):
        if t < cooldown_until:
            V[t] = v_reset
            cooldown_trace[t] = 1
        elif door[t] == 1:  # Open
            dV = (-V[t - 1] / tau_m) * dt + bias_current
            V[t] = V[t - 1] + dV
            if V[t] >= v_th:
                alert[t] = 1
                V[t] = v_reset
                cooldown_until = t + int(cooldown_s / dt)
        else:  # Closed
            V[t] = v_reset

    return V, alert, cooldown_trace


def compute_ground_truth_timeouts(segments, open_timeout_s, dt):
    """
    Compute when each door segment should trigger a timeout alert.

    Returns
    -------
    timeout_points: list[int]
    timeout_segments: list[int]
    """
    timeout_points = []
    timeout_segments = []

    for seg_idx, (start, end) in enumerate(segments):
        duration = (end - start) * dt
        if duration > open_timeout_s:
            timeout_idx = start + int(open_timeout_s / dt)
            if timeout_idx < end:
                timeout_points.append(timeout_idx)
                timeout_segments.append(seg_idx)

    return timeout_points, timeout_segments


def write_csv(path: Path, headers, rows_iterable):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows_iterable:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Generate example door/SNN data")
    parser.add_argument('--scenario', choices=['mixed', 'stochastic'], default='mixed')
    parser.add_argument('--dt', type=float, default=0.1, help='time step in seconds')
    parser.add_argument('--sim-minutes', type=float, default=30.0, help='total simulation minutes')
    parser.add_argument('--outdir', type=str, default='example_data', help='output directory')
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Parameters (aligned with the notebook)
    dt = args.dt
    sim_minutes = args.sim_minutes
    T = int(sim_minutes * 60 / dt)

    open_timeout_s = 20.0
    cooldown_s = 10.0
    tau_m = 10.0
    v_th = 1.0
    v_reset = 0.0
    bias_current = v_th * dt / open_timeout_s

    # Data generation
    door, segments = generate_door_stream(T, dt, scenario=args.scenario)
    V, alert, cooldown_trace = simulate_lif_timer(
        door, dt, tau_m, v_th, v_reset, bias_current, cooldown_s
    )
    timeout_points, timeout_segments = compute_ground_truth_timeouts(
        segments, open_timeout_s, dt
    )

    # Prepare outputs
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    time_s = np.arange(T) * dt
    minute = time_s / 60.0

    # Map each time index to segment id (-1 when closed)
    seg_id = np.full(T, -1, dtype=np.int32)
    for idx, (s, e) in enumerate(segments):
        seg_id[s:e] = idx

    # Write door events with segment ids
    door_rows = (
        (float(t), float(m), int(d), int(sid))
        for t, m, d, sid in zip(time_s, minute, door, seg_id)
    )
    write_csv(
        outdir / 'door_events.csv',
        headers=['time_s', 'minute', 'door', 'segment_id'],
        rows_iterable=door_rows,
    )

    # Write simulation outputs
    sim_rows = (
        (float(t), float(m), int(d), float(v), int(a), int(cd))
        for t, m, d, v, a, cd in zip(time_s, minute, door, V, alert, cooldown_trace)
    )
    write_csv(
        outdir / 'simulation_output.csv',
        headers=['time_s', 'minute', 'door', 'V', 'alert', 'cooldown'],
        rows_iterable=sim_rows,
    )

    # Write ground-truth timeout points
    timeout_rows = (
        (float(time_s[tp]), float(minute[tp]), int(ts))
        for tp, ts in zip(timeout_points, timeout_segments)
    )
    write_csv(
        outdir / 'timeouts.csv',
        headers=['time_s', 'minute', 'segment_index'],
        rows_iterable=timeout_rows,
    )

    # Write metadata
    metadata = {
        'scenario': args.scenario,
        'dt': dt,
        'sim_minutes': sim_minutes,
        'T': T,
        'parameters': {
            'open_timeout_s': open_timeout_s,
            'cooldown_s': cooldown_s,
            'tau_m': tau_m,
            'v_th': v_th,
            'v_reset': v_reset,
            'bias_current': bias_current,
        },
        'counts': {
            'segments': len(segments),
            'timeout_segments': len(timeout_segments),
            'alerts': int(np.sum(alert)),
        },
    }
    with (outdir / 'metadata.json').open('w') as f:
        json.dump(metadata, f, indent=2)

    # Console summary
    print("Generated Example Data:")
    print(f"  Scenario: {args.scenario}")
    print(f"  Duration: {sim_minutes} min, dt={dt}s, steps={T}")
    print(f"  Segments: {len(segments)} | Timeouts: {len(timeout_segments)} | Alerts: {int(np.sum(alert))}")
    print(f"  Output directory: {outdir.resolve()}")
    print("  Files:")
    print("    - door_events.csv")
    print("    - simulation_output.csv")
    print("    - timeouts.csv")
    print("    - metadata.json")


if __name__ == '__main__':
    main()
