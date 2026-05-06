import matplotlib.pyplot as plt
import pandas as pd
import ffmpeg
import os
import json
import plotly.express as px
import zipfile
import io
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata

def get_total_duration(folder_path: str):
    """Derive source video duration from raw videos.

    Layout: <frames_root>/<video_dir>/

    Returns (total_hours, "HH:MM:SS")."""
    total_seconds = 0.0

    for fn in os.listdir(folder_path):
        if not fn.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue
        full_path = os.path.join(folder_path, fn)
        parser = createParser(full_path)
        if parser is None:
            print(f"Could not parse: {fn}")
            continue
        metadata = extractMetadata(parser)
        if metadata and metadata.has("duration"):
            total_seconds += metadata.get("duration").total_seconds()

    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = int(total_seconds % 60)
    return total_seconds / 3600, f"{h:02d}:{m:02d}:{s:02d}"

def get_total_duration_from_frames(frames_root: str, extract_fps: int = 1):
    """Derive source video duration from extracted frames (no raw videos needed).

    Layout: <frames_root>/<video_dir>/<frame_file>. Cholec80 frames are
    extracted at 1 fps from 25 fps source, so total_frames / extract_fps
    gives seconds of source video.

    Returns (total_hours, "HH:MM:SS").
    """
    total_frames = 0
    for vid in os.scandir(frames_root):
        if vid.is_dir():
            total_frames += sum(
                1 for f in os.scandir(vid.path)
                if f.is_file() and not f.name.startswith(".")
            )
    total_seconds = total_frames / extract_fps
    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = int(total_seconds % 60)
    return total_seconds / 3600, f"{h:02d}:{m:02d}:{s:02d}"

def count_frames(frames_root):
    """Count files in <frames_root>/<video>/<frame_file> (any extension)."""
    return sum(
        sum(
            1 for f in os.scandir(v.path)
            if f.is_file() and not f.name.startswith(".")
        )
        for v in os.scandir(frames_root) if v.is_dir()
    )


if __name__ == "__main__":
    ROOTS = {
        "PitVis":    {"videos": "./data/PitVis",    "frames": "./data/PitViS/frames",    "type": "Pretraining"},
        "CATARACTS": {"videos": "./data/CATARACTS", "frames": "./data/CATARACTS/frames", "type": "Pretraining"},
        "JIGSAWS":   {"videos": "./data/JIGSAWS",   "frames": "./data/JIGSAWS/frames",   "type": "Pretraining"},
        "Cholec80":  {"videos": None,               "frames": "./data/cholec80/frames",  "type": "Evaluation"},
    }

    records = []
    for name, cfg in ROOTS.items():
        n_frames = count_frames(cfg["frames"])
        if cfg["videos"]:
            hours, _ = get_total_duration(cfg["videos"])
        else:
            hours, _ = get_total_duration_from_frames(cfg["frames"], extract_fps=1)
        records.append({
            "Dataset": name,
            "Hours":   hours,
            "Frames":  n_frames,
            "Type":    cfg["type"],
        })

    df = pd.DataFrame(records).sort_values("Frames", ascending=False).reset_index(drop=True)

    # Stacked facets: total frames (top) and total hours (bottom) per dataset.
    long_df = df.melt(
        id_vars=["Dataset", "Type"],
        value_vars=["Frames"],
        var_name="Metric",
        value_name="Value",
    )
    long_df["Label"] = [
        f"{int(v):,}" if m == "Frames" else f"{v:.1f}h"
        for v, m in zip(long_df["Value"], long_df["Metric"])
    ]

    fig = px.bar(
        long_df,
        x="Dataset",
        y="Value",
        color="Type",
        facet_row="Frames",
        facet_row_spacing=0.18,
        text="Label",
        color_discrete_map={"Pretraining": "#636EFA", "Evaluation": "#EF553B"},
        template="plotly_white",
    )

    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="Datasets"
            )
        ),
        yaxis=dict(
            title=dict(
                text="Frames"
            )
        ),
        legend=dict(
            title=dict(
                text="Usage"
            )
        ),
        font=dict(
            family="Courier New, monospace",
            size=24,
        )

    )

    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        marker_line_color="rgb(8,48,107)",
        marker_line_width=1.2,
        opacity=0.9,
    )

    # Horizontal divider between the two facets (paper coords; midpoint of the gap).

    fig.show(renderer="browser")
    fig.write_html("dataset_plotly.html")