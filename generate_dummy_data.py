"""
generate_dummy_data.py
======================
Generates synthetic CSI data that matches your PostgreSQL schema and
produces realistic per-class waveform patterns.

Each movement class has physically motivated signal characteristics:

  Standing  → Stable, low-variance amplitude. Person is stationary so
               multipath components barely change. Phase is near-constant.

  Sitting   → Very low variance, slightly lower amplitude than Standing
               because the person is smaller/lower. Phase nearly flat.

  Walking   → Periodic amplitude oscillations (~1-2 Hz) caused by arm
               swing and leg movement disturbing the multipath. Phase
               oscillates in sync.

  Running   → Higher-frequency oscillations (~3-5 Hz), much wider
               amplitude swings. Phase is chaotic due to fast motion.

  Falling   → Short sudden amplitude spike (initial body movement) followed
               by a sharp drop to near-zero (person is now on the floor,
               minimal movement). Phase becomes very noisy during the spike.

OUTPUT
──────
  1. data/raw/csi_data.csv          — ready for python train.py
  2. scripts/insert_into_postgres.sql — optional: load into your real DB

USAGE
─────
  python scripts/generate_dummy_data.py
  python scripts/generate_dummy_data.py --sessions 200 --packets 60
  python scripts/generate_dummy_data.py --output my_data.csv --no-sql
  python scripts/generate_dummy_data.py --seed 99
"""

import argparse
import csv
import math
import os
import random
import uuid
from datetime import datetime, timedelta, timezone

# ── Project root (one level above this script) ────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "data", "raw")
SQL_DIR   = os.path.join(ROOT, "scripts")

# ── Movement class definitions ────────────────────────────────────────────────
# Each entry controls how the waveform generator behaves for that class.
#
# Keys
# ─────
#   base_amp   : mean amplitude across subcarriers
#   amp_std    : standard deviation of the base amplitude noise
#   phase_std  : standard deviation of phase noise
#   osc_freq   : oscillation frequency in Hz (0 = no oscillation)
#   osc_amp    : oscillation amplitude added on top of base_amp
#   spike      : if True, add a single-packet amplitude spike (Falling only)
#   rssi_mean  : typical RSSI dBm value
#   rssi_std   : RSSI jitter

MOVEMENT_PROFILES = {
    "Standing": {
        "base_amp"  : 13.0,
        "amp_std"   : 0.8,
        "phase_std" : 0.15,
        "osc_freq"  : 0.0,
        "osc_amp"   : 0.0,
        "spike"     : False,
        "rssi_mean" : -48.0,
        "rssi_std"  : 1.5,
        "description": "Stable, low-variance — person not moving",
    },
    "Sitting": {
        "base_amp"  : 10.5,
        "amp_std"   : 0.5,
        "phase_std" : 0.10,
        "osc_freq"  : 0.0,
        "osc_amp"   : 0.0,
        "spike"     : False,
        "rssi_mean" : -50.0,
        "rssi_std"  : 1.2,
        "description": "Very stable — person seated and still",
    },
    "Walking": {
        "base_amp"  : 14.5,
        "amp_std"   : 2.5,
        "phase_std" : 0.60,
        "osc_freq"  : 1.5,    # Hz — one full stride ~0.7 s
        "osc_amp"   : 4.0,
        "spike"     : False,
        "rssi_mean" : -47.0,
        "rssi_std"  : 2.5,
        "description": "Periodic ~1.5 Hz oscillation — arm/leg swing",
    },
    "Running": {
        "base_amp"  : 18.0,
        "amp_std"   : 6.0,
        "phase_std" : 1.20,
        "osc_freq"  : 3.5,    # Hz — fast stride rate
        "osc_amp"   : 12.0,
        "spike"     : False,
        "rssi_mean" : -45.0,
        "rssi_std"  : 4.0,
        "description": "High-freq, high-amp oscillation — fast body movement",
    },
    "Falling": {
        "base_amp"  : 8.0,    # resting (post-fall on floor)
        "amp_std"   : 1.0,
        "phase_std" : 0.20,
        "osc_freq"  : 0.0,
        "osc_amp"   : 0.0,
        "spike"     : True,   # abrupt spike at the moment of falling
        "rssi_mean" : -52.0,
        "rssi_std"  : 2.0,
        "description": "Spike then near-silence — sudden fall then floor",
    },
}

# Available WiFi channel configs matching your schema
CHANNEL_CONFIGS = [
    {"center_frequency_ghz": 2.400, "channel_number":  6, "bandwidth_mhz": 20, "subcarriers": 56},
    {"center_frequency_ghz": 2.400, "channel_number": 11, "bandwidth_mhz": 20, "subcarriers": 56},
    {"center_frequency_ghz": 5.000, "channel_number": 36, "bandwidth_mhz": 40, "subcarriers": 114},
    {"center_frequency_ghz": 5.000, "channel_number": 40, "bandwidth_mhz": 40, "subcarriers": 114},
]

# Fixed UUIDs from your schema (devices / rooms / users)
DEVICE_IDS = [
    "40000000-0000-0000-0000-000000000001",
    "40000000-0000-0000-0000-000000000002",
    "40000000-0000-0000-0000-000000000003",
    "40000000-0000-0000-0000-000000000004",
    "40000000-0000-0000-0000-000000000005",
]
ROOM_IDS = [
    "30000000-0000-0000-0000-000000000001",
    "30000000-0000-0000-0000-000000000002",
    "30000000-0000-0000-0000-000000000003",
    "30000000-0000-0000-0000-000000000004",
    "30000000-0000-0000-0000-000000000005",
]
USER_IDS = [
    "20000000-0000-0000-0000-000000000002",
    "20000000-0000-0000-0000-000000000003",
    "20000000-0000-0000-0000-000000000005",
]
MODEL_VERSION_ID = "A0000000-0000-0000-0000-000000000002"  # RF v1.1 Active


# ═════════════════════════════════════════════════════════════════════════════
# Waveform generators
# ═════════════════════════════════════════════════════════════════════════════

def _generate_amplitude(
    profile: dict,
    n_subcarriers: int,
    packet_index: int,
    sample_rate_hz: int,
    is_spike_packet: bool,
    rng: random.Random,
) -> list[float]:
    """
    Generate one amplitude waveform (one packet) for a given movement class.

    The waveform represents signal strength across subcarriers, not time —
    but the oscillation is modulated by the packet timestamp to make
    consecutive packets look realistic.
    """
    t = packet_index / sample_rate_hz   # seconds since session start

    # Base amplitude across subcarriers with Gaussian noise
    base   = profile["base_amp"]
    std    = profile["amp_std"]

    # Oscillation component (walking / running)
    osc = 0.0
    if profile["osc_freq"] > 0:
        osc = profile["osc_amp"] * math.sin(2 * math.pi * profile["osc_freq"] * t)

    # For a fall: first packet is a spike, subsequent packets are quiet
    if is_spike_packet:
        base = base + rng.uniform(25, 40)   # sudden high-energy burst
        std  = std * 6                       # very noisy

    # Generate per-subcarrier values
    # Real CSI amplitude is smooth across adjacent subcarriers (correlation),
    # so we add a slow-varying component on top of the iid noise.
    slow_component = [
        2.0 * math.sin(2 * math.pi * i / n_subcarriers * rng.uniform(2, 5))
        for i in range(n_subcarriers)
    ]

    values = []
    for i in range(n_subcarriers):
        v = base + osc + slow_component[i] + rng.gauss(0, std)
        v = max(0.1, v)   # amplitude is always positive
        values.append(round(v, 4))

    return values


def _generate_phase(
    profile: dict,
    n_subcarriers: int,
    packet_index: int,
    sample_rate_hz: int,
    is_spike_packet: bool,
    rng: random.Random,
) -> list[float]:
    """
    Generate one phase waveform.

    Real CSI phase is a linear ramp across subcarriers (due to timing offset)
    with small perturbations.  We simulate this with a random slope + noise.
    """
    t         = packet_index / sample_rate_hz
    phase_std = profile["phase_std"]

    # Linear phase ramp across subcarriers (slope changes slowly with motion)
    base_slope = rng.gauss(0.06, 0.01)      # radians per subcarrier
    if is_spike_packet:
        phase_std = phase_std * 5            # chaotic during fall

    values = []
    for i in range(n_subcarriers):
        # Linear ramp + motion-induced oscillation + noise
        ramp = base_slope * (i - n_subcarriers / 2)
        if profile["osc_freq"] > 0:
            ramp += 0.3 * math.sin(2 * math.pi * profile["osc_freq"] * t + i * 0.1)
        v = ramp + rng.gauss(0, phase_std)
        # Clamp to [-π, π]
        v = max(-math.pi, min(math.pi, v))
        values.append(round(v, 4))

    return values


def _pg_array(values: list[float]) -> str:
    """Format a list as a PostgreSQL array literal: {v1,v2,...}"""
    return "{" + ",".join(str(v) for v in values) + "}"


# ═════════════════════════════════════════════════════════════════════════════
# Session generator
# ═════════════════════════════════════════════════════════════════════════════

def generate_session(
    movement_label: str,
    packets_per_session: int,
    device_idx: int,
    session_start: datetime,
    sample_rate_hz: int,
    rng: random.Random,
) -> list[dict]:
    """
    Generate all CSI packets for one session of a given movement class.

    Returns a list of row dicts ready for CSV output.
    """
    profile    = MOVEMENT_PROFILES[movement_label]
    channel    = CHANNEL_CONFIGS[device_idx % len(CHANNEL_CONFIGS)]
    n_sub      = channel["subcarriers"]

    session_id   = str(uuid.uuid4())
    device_id    = DEVICE_IDS[device_idx % len(DEVICE_IDS)]
    room_id      = ROOM_IDS[device_idx   % len(ROOM_IDS)]

    # For Falling: the spike packet is somewhere in the first third
    spike_idx = rng.randint(2, max(3, packets_per_session // 3)) \
        if profile["spike"] else -1

    rssi_base = rng.gauss(profile["rssi_mean"], profile["rssi_std"])

    rows = []
    for pkt_idx in range(packets_per_session):
        is_spike = (pkt_idx == spike_idx)

        amp   = _generate_amplitude(profile, n_sub, pkt_idx, sample_rate_hz, is_spike, rng)
        phase = _generate_phase(    profile, n_sub, pkt_idx, sample_rate_hz, is_spike, rng)

        capture_time = session_start + timedelta(seconds=pkt_idx / sample_rate_hz)
        rssi         = round(rssi_base + rng.gauss(0, 0.5), 2)

        rows.append({
            "session_id"           : session_id,
            "device_id"            : device_id,
            "room_id"              : room_id,
            "capture_time"         : capture_time.isoformat(),
            "center_frequency_ghz" : channel["center_frequency_ghz"],
            "channel_number"       : channel["channel_number"],
            "bandwidth_mhz"        : channel["bandwidth_mhz"],
            "subcarriers"          : n_sub,
            "sample_rate_hz"       : sample_rate_hz,
            "rssi_dbm"             : rssi,
            "amplitude_waveform"   : _pg_array(amp),
            "phase_waveform"       : _pg_array(phase),
            "predicted_movement"   : movement_label,
        })

    return rows


# ═════════════════════════════════════════════════════════════════════════════
# SQL INSERT generator  (optional)
# ═════════════════════════════════════════════════════════════════════════════

def _generate_sql(rows: list[dict], out_path: str):
    """
    Write a .sql file with INSERT statements for csi_metadata + ml_predictions.
    Useful if you want to populate your real database.
    """
    meta_id_map = {}   # row_index → csi_metadata_id

    lines = [
        "-- Auto-generated dummy CSI data",
        "-- Generated by scripts/generate_dummy_data.py",
        "-- Import with:  psql -d your_db -f insert_into_postgres.sql",
        "",
        "BEGIN;",
        "",
        "-- ── csi_metadata ────────────────────────────────────────────",
    ]

    for i, row in enumerate(rows):
        meta_id   = str(uuid.uuid4())
        meta_id_map[i] = meta_id

        # Pick a session UUID from the row (session_id is already in the row)
        # We need device_id and room_id from devices/rooms (fixed in your schema)
        device_uuid = row["device_id"]
        room_uuid   = row["room_id"]
        session_uuid = row["session_id"]

        lines.append(
            f"INSERT INTO csi_metadata "
            f"(csi_metadata_id, session_id, device_id, room_id, capture_time, "
            f"center_frequency_ghz, channel_number, bandwidth_mhz, subcarriers, "
            f"sample_rate_hz, rssi_dbm, amplitude_waveform, phase_waveform) VALUES ("
            f"'{meta_id}', '{session_uuid}', '{device_uuid}', '{room_uuid}', "
            f"'{row['capture_time']}', "
            f"{row['center_frequency_ghz']}, {row['channel_number']}, "
            f"{row['bandwidth_mhz']}, {row['subcarriers']}, {row['sample_rate_hz']}, "
            f"{row['rssi_dbm']}, "
            f"'{row['amplitude_waveform']}', "
            f"'{row['phase_waveform']}'"
            f");"
        )

    lines += [
        "",
        "-- ── ml_predictions ──────────────────────────────────────────",
    ]

    for i, row in enumerate(rows):
        pred_id  = str(uuid.uuid4())
        meta_id  = meta_id_map[i]
        movement = row["predicted_movement"]
        # Use a realistic confidence: higher for clear movements, lower for ambiguous
        confidence_map = {
            "Standing": (92, 99),
            "Sitting" : (88, 97),
            "Walking" : (85, 96),
            "Running" : (87, 99),
            "Falling" : (90, 99),
        }
        lo, hi = confidence_map.get(movement, (80, 95))
        confidence = round(random.uniform(lo, hi), 2)

        lines.append(
            f"INSERT INTO ml_predictions "
            f"(prediction_id, csi_metadata_id, model_version_id, "
            f"predicted_movement, confidence_score, prediction_time) VALUES ("
            f"'{pred_id}', '{meta_id}', '{MODEL_VERSION_ID}', "
            f"'{movement}', {confidence}, '{row['capture_time']}'::timestamptz + interval '2 seconds'"
            f");"
        )

    lines += ["", "COMMIT;", ""]

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[SQL]  ✓ INSERT script → {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic CSI dummy data for ML training."
    )
    parser.add_argument(
        "--sessions",  type=int, default=60,
        help="Sessions PER movement class (default: 60 → 300 total sessions)",
    )
    parser.add_argument(
        "--packets",   type=int, default=50,
        help="CSI packets per session (default: 50)",
    )
    parser.add_argument(
        "--rate",      type=int, default=10,
        help="Simulated sample_rate_hz (default: 10)",
    )
    parser.add_argument(
        "--output",    type=str, default=None,
        help="Output CSV path (default: data/raw/csi_data.csv)",
    )
    parser.add_argument(
        "--no-sql",    action="store_true",
        help="Skip generating the PostgreSQL INSERT script",
    )
    parser.add_argument(
        "--seed",      type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    random.seed(args.seed)   # for SQL confidence values

    labels   = list(MOVEMENT_PROFILES.keys())
    n_per    = args.sessions
    n_total  = n_per * len(labels)

    out_csv  = args.output or os.path.join(DATA_DIR, "csi_data.csv")
    out_sql  = os.path.join(SQL_DIR, "insert_into_postgres.sql")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    print("=" * 60)
    print("  CSI Dummy Data Generator")
    print("=" * 60)
    print(f"  Classes     : {labels}")
    print(f"  Sessions    : {n_per} per class  →  {n_total} total")
    print(f"  Packets     : {args.packets} per session")
    print(f"  Total rows  : {n_total * args.packets:,}")
    print(f"  Output CSV  : {out_csv}")
    if not args.no_sql:
        print(f"  Output SQL  : {out_sql}")
    print(f"  Seed        : {args.seed}")
    print()

    all_rows = []

    # Base timestamp: sessions spread over the past 7 days
    base_time = datetime(2026, 8, 18, 8, 0, 0, tzinfo=timezone.utc)
    session_gap = timedelta(minutes=5)
    current_time = base_time

    for label in labels:
        print(f"  Generating '{label}' ({n_per} sessions × {args.packets} packets) …")
        for sess_i in range(n_per):
            device_idx = sess_i % len(DEVICE_IDS)
            session_rows = generate_session(
                movement_label      = label,
                packets_per_session = args.packets,
                device_idx          = device_idx,
                session_start       = current_time,
                sample_rate_hz      = args.rate,
                rng                 = rng,
            )
            all_rows.extend(session_rows)
            current_time += session_gap

    # Shuffle so the CSV isn't grouped by label (avoids accidental ordering bias)
    # But keep within-session order intact by shuffling at session level first
    print("\n  Shuffling session order …")
    # Group rows by session_id, shuffle sessions, then flatten
    sessions_map: dict[str, list[dict]] = {}
    for row in all_rows:
        sessions_map.setdefault(row["session_id"], []).append(row)

    session_list = list(sessions_map.values())
    rng.shuffle(session_list)

    shuffled_rows = [row for session in session_list for row in session]

    # Write CSV
    fieldnames = [
        "session_id", "device_id", "room_id", "capture_time",
        "center_frequency_ghz", "channel_number", "bandwidth_mhz",
        "subcarriers", "sample_rate_hz", "rssi_dbm",
        "amplitude_waveform", "phase_waveform", "predicted_movement",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(shuffled_rows)

    print(f"\n[CSV]  ✓ {len(shuffled_rows):,} rows → {out_csv}")

    # Label distribution check
    from collections import Counter
    dist = Counter(r["predicted_movement"] for r in shuffled_rows)
    print("\n  Label distribution:")
    for label, count in sorted(dist.items()):
        print(f"    {label:<12s}: {count:>6,} packets  ({n_per} sessions)")

    # Write SQL
    if not args.no_sql:
        print("\n  Generating PostgreSQL INSERT script …")
        _generate_sql(shuffled_rows, out_sql)

    print("\n" + "=" * 60)
    print("  Done!  Next steps:")
    print()
    print("  1. Train the ML models:")
    print("       python train.py")
    print()
    print("  2. (Optional) Load into your database:")
    if not args.no_sql:
        print("       psql -d your_db -f scripts/insert_into_postgres.sql")
    print()
    print("  ⚠  Remember: this is SYNTHETIC data.")
    print("     Replace with real CSI captures before deploying.")
    print("=" * 60)


if __name__ == "__main__":
    main()
