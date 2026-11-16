# CrackSealingRobot

An automated crack-sealing robot with RGB-D perception, crack quantification, trajectory planning, and coordinated robotic arm and pump control.

## Installation and Quick Start

Requires Python ≥ 3.10. Run from the project root:

```bash
python -m pip install -e .
python pipeline.py
```

Install the optional hardware dependencies when using the camera or pump. The Unitree SDK must be installed separately:

```bash
python -m pip install -e '.[hardware]'
```

## Directory Structure

```text
CrackSealingRobot/
├── perception/          # Perception and inference
│   ├── best.pt          # Model weights
│   └── data/            # Sample data
│       ├── rgb/         # RGB images with matching IDs
│       ├── depth/       # Raw depth maps with matching IDs
│       └── mask/        # Annotation masks with matching IDs
├── quantification/      # Crack skeletons and physical quantification
├── trajectory_planning/ # Crack trajectory planning
├── control/             # Calibration, robotic arm and pump control
├── contracts.py         # Shared data structures
├── pipeline.py          # Unified entry point
├── config.yaml          # Runtime configuration
└── pyproject.toml       # Project dependencies
```

## Running the Pipeline

```bash
python pipeline.py --config config.yaml
```

Select the input source and execution mode in `config.yaml`:

- `input.source`: `offline` (local data) or `d405` (camera).
- `input.root`: defaults to `perception/data/`; RGB images and raw depth maps are paired by filename.
- `control.mode`: `plan` (planning), `simulate` (simulation), or `real` (hardware execution).

By default, the pipeline processes the RGB-D images in the data directory through the planning stage. Results are saved to `control/runs/run_<timestamp>/`, including masks, overlays, skeletons, quantification results, and routes. Configuration paths are resolved relative to the configuration file.
