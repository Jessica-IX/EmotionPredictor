#!/usr/bin/env python3
"""
Preprocess hana sessions into image + emotion probability vector pairs.

For every batch directory found under the provided sessions root, this script:
  1. Reads the ROS2 bag stored in data/data_0.db3 (sqlite backend).
  2. Extracts all labels from the /emotion_estimate topic (std_msgs/String).
  3. Counts how often each tracked emotion appears in that batch and converts
     the counts into a probability distribution.
  4. Copies the corresponding canvas.png into the output folder, next to a JSON
     file containing the label order, counts, and probabilities.

The tracked emotions can be provided explicitly (via --labels or --label-file)
or automatically discovered by scanning all batches and picking the top-N
labels (default 9). Automatic discovery guarantees the same ordering is used
across every batch output.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

# Standard emotion label order and mapping from data labels to output labels
EMOTION_LABEL_ORDER = [
    "amusement",
    "awe",
    "contentment",
    "excitement",
    "anger",
    "disgust",
    "fear",
    "sadness",
    "something else",
]

# Mapping from labels as they appear in the data to normalized output labels
LABEL_MAPPING = {
    "Amusement": "amusement",
    "Awe": "awe",
    "Contentment": "contentment",
    "Excitement": "excitement",
    "Anger": "anger",
    "Disgust": "disgust",
    "Fear": "fear",
    "Sadness": "sadness",
    "Something-else": "something else",
    "Something else": "something else",
    # Also handle lowercase variants
    "amusement": "amusement",
    "awe": "awe",
    "contentment": "contentment",
    "excitement": "excitement",
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "sadness": "sadness",
    "something-else": "something else",
    "something else": "something else",
}


def normalize_label(label: str) -> str:
    """Normalize emotion labels from data format to output format."""
    return LABEL_MAPPING.get(label, label.lower())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create batch-wise folders containing canvas.png and the "
            "associated emotion probability vector derived from /emotion_estimate."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root directory containing hana-session-* folders (default: current dir).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("processed"),
        help="Directory where processed batch folders will be written.",
    )
    parser.add_argument(
        "--sessions",
        nargs="*",
        help="Optional subset of session folder names to process (e.g. hana-session-1).",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Explicit list of tracked emotions (overrides automatic discovery).",
    )
    parser.add_argument(
        "--label-file",
        type=Path,
        help=(
            "Path to a text or JSON file containing the tracked emotion labels. "
            "Text files are read line-by-line; JSON files must contain an array of strings."
        ),
    )
    parser.add_argument(
        "--num-labels",
        type=int,
        default=9,
        help="Number of labels to keep when discovering automatically (default: 9).",
    )
    parser.add_argument(
        "--topic",
        default="/emotion_estimate",
        help="Topic name that carries the discrete emotion labels.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect batches and label distribution without writing outputs.",
    )
    return parser.parse_args()


def load_labels_from_file(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Label file {path} does not exist")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Label file {path} is empty")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise ValueError(f"JSON label file must contain an array of strings: {path}")
        return [label.strip() for label in data if label.strip()]
    # Fallback: newline-separated labels
    return [line.strip() for line in text.splitlines() if line.strip()]


def iter_sessions(root: Path, session_filters: Sequence[str] | None) -> Iterator[Path]:
    if session_filters:
        for session_name in session_filters:
            session_dir = root / session_name
            if session_dir.is_dir():
                yield session_dir
            else:
                print(
                    f"[WARN] Requested session {session_name} does not exist under {root}",
                    file=sys.stderr,
                )
        return
    for session_dir in sorted(root.glob("hana-session-*")):
        if session_dir.is_dir():
            yield session_dir


def iter_batches(session_dir: Path) -> Iterator[Path]:
    for batch_dir in sorted(session_dir.glob("batch_*")):
        if batch_dir.is_dir():
            yield batch_dir


def get_topic_id(conn: sqlite3.Connection, topic_name: str) -> int | None:
    cursor = conn.execute("SELECT id FROM topics WHERE name = ?", (topic_name,))
    row = cursor.fetchone()
    return row[0] if row else None


def parse_cdr_string(blob: bytes) -> str:
    """
    Parse a ROS2 std_msgs/msg/String serialized payload (little endian CDR).
    Layout: <encapsulation:uint32><length:uint32><bytes...>.
    """
    data = memoryview(blob)
    if len(data) < 8:
        return ""
    length = struct.unpack_from("<I", data, 4)[0]
    raw = data[8 : 8 + length]
    return raw.tobytes().decode("utf-8", errors="ignore").rstrip("\x00")


def read_emotion_labels(db_path: Path, topic_name: str, normalize: bool = True) -> List[str] | None:
    emitted: List[str] = []
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            topic_id = get_topic_id(conn, topic_name)
            if topic_id is None:
                return emitted
            for (blob,) in conn.execute(
                "SELECT data FROM messages WHERE topic_id = ?", (topic_id,)
            ):
                label = parse_cdr_string(blob)
                if normalize:
                    label = normalize_label(label)
                emitted.append(label)
    except sqlite3.Error as exc:
        print(f"[WARN] Failed to read {db_path}: {exc}", file=sys.stderr)
        return None
    return emitted


def discover_labels(
    sessions_root: Path,
    session_filters: Sequence[str] | None,
    topic_name: str,
    num_labels: int,
) -> List[str]:
    global_counts: Counter[str] = Counter()
    for session_dir in iter_sessions(sessions_root, session_filters):
        for batch_dir in iter_batches(session_dir):
            db_path = batch_dir / "data" / "data_0.db3"
            if not db_path.exists():
                continue
            labels = read_emotion_labels(db_path, topic_name)
            if labels is None:
                continue
            global_counts.update(labels)
    if not global_counts:
        raise RuntimeError("No emotion labels found. Is the topic name correct?")
    most_common = [label for label, _ in global_counts.most_common(num_labels)]
    if len(most_common) < num_labels:
        print(
            f"[INFO] Only found {len(most_common)} unique labels; "
            f"probability vectors will use that length.",
            file=sys.stderr,
        )
    return most_common


def ensure_output_tree(output_root: Path, session_dir: Path, batch_dir: Path) -> Path:
    target = output_root / session_dir.name / batch_dir.name
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_batch_artifacts(
    batch_dir: Path,
    dest_dir: Path,
    label_order: Sequence[str],
    counts: Dict[str, int],
) -> None:
    total = sum(counts.values())
    ordered_counts = [counts.get(label, 0) for label in label_order]
    probabilities = [
        (count / total) if total > 0 else 0.0 for count in ordered_counts
    ]
    payload = {
        "session": batch_dir.parent.name,
        "batch": batch_dir.name,
        "labels": list(label_order),
        "counts": ordered_counts,
        "probabilities": probabilities,
        "total_samples": total,
    }
    (dest_dir / "emotion_probabilities.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    canvas_src = batch_dir / "canvas.png"
    if canvas_src.exists():
        dest_canvas = dest_dir / "canvas.png"
        dest_canvas.write_bytes(canvas_src.read_bytes())
    else:
        print(f"[WARN] Missing canvas.png in {batch_dir}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    sessions_root = args.root.resolve()
    output_root = args.output.resolve()
    if not sessions_root.exists():
        raise SystemExit(f"Root directory {sessions_root} does not exist")

    if args.labels:
        label_order = [label.strip() for label in args.labels if label.strip()]
    elif args.label_file:
        label_order = load_labels_from_file(args.label_file)
    else:
        # Use the standard emotion label order as default
        label_order = EMOTION_LABEL_ORDER.copy()
        print(f"[INFO] Using standard emotion label order: {label_order}")
    if not label_order:
        raise SystemExit("No labels were specified or discovered.")

    print(f"[INFO] Using label order: {label_order}")
    summary = defaultdict(int)
    processed_batches = 0

    for session_dir in iter_sessions(sessions_root, args.sessions):
        for batch_dir in iter_batches(session_dir):
            db_path = batch_dir / "data" / "data_0.db3"
            if not db_path.exists():
                print(f"[WARN] Skipping {batch_dir}: missing {db_path.name}", file=sys.stderr)
                continue
            labels = read_emotion_labels(db_path, args.topic)
            if labels is None:
                print(f"[WARN] Skipping {batch_dir}: unable to read emotion labels", file=sys.stderr)
                continue
            batch_counts = Counter(labels)
            for label in label_order:
                summary[label] += batch_counts.get(label, 0)
            if args.dry_run:
                continue
            dest_dir = ensure_output_tree(output_root, session_dir, batch_dir)
            write_batch_artifacts(batch_dir, dest_dir, label_order, batch_counts)
            processed_batches += 1

    if args.dry_run:
        print("[DRY-RUN] Skipped writing outputs. Aggregate counts:")
    else:
        print(f"[DONE] Wrote outputs for {processed_batches} batches.")
    for label in label_order:
        print(f"  {label}: {summary[label]}")


if __name__ == "__main__":
    main()

