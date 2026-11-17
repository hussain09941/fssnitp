# Few-Shot Semantic Segmentation — Pascal VOC 2012

A modular local Python project for 4-fold few-shot semantic segmentation.

## Project structure

```text
FewShot-Semantic-Segmentation-Local/
├── data/                         # Pascal VOC dataset
├── checkpoints/                  # Saved fold model weights
├── results/visualizations/       # Generated visualizations
├── src/fewshot_segmentation/
│   ├── config.py                 # Settings, device, classes and folds
│   ├── dataset.py                # VOC discovery, masks and image loading
│   ├── models.py                 # Multi-scale ResNet50 backbone
│   ├── prototypes.py             # FG/BG and K-shot prototypes
│   ├── losses.py                 # BCE, Dice, prediction and metrics
│   ├── episodes.py               # Few-shot episode creation
│   ├── training.py               # One training episode
│   ├── inference.py              # Refinement and inference
│   ├── evaluation.py             # Fold/model evaluation helpers
│   ├── diagnostics.py            # Same-image diagnostic
│   ├── visualization.py          # Episode visualization
│   └── pipeline.py               # End-to-end orchestration
├── requirements.txt
├── .env.example
└── run.py
```

## Setup (macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Put Pascal VOC 2012 under `data/` in the standard layout. The dataset root must contain `JPEGImages/` and `SegmentationClass/`. If it is elsewhere, set `VOC_DATASET_PATH` to the dataset directory.

Run:

```bash
python run.py
```

Quick smoke test (small number of episodes):

```bash
FEWSHOT_EPISODES_PER_FOLD=2 FEWSHOT_EPISODES_PER_TEST_FOLD=1 python run.py
```

The model uses CUDA when available, Apple Silicon MPS when available, otherwise CPU. The first model creation may download pretrained ResNet50 weights.

**Note:** This is a code-organization refactor of the supplied implementation. The complete training run has not been executed here; run the smoke test first on your machine.
