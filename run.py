
import sys
from pathlib import Path

# Add src directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

from fewshot_segmentation.pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline()
