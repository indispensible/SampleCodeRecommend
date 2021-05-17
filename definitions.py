import os
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root

# project data
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# the output dir
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')

Doc_Path = Path(DATA_DIR) / "doc" / "jdk8.v1.dc"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root

# the data dir
DATA_DIR = os.path.join(ROOT_DIR, 'data')
# the data dir
LABEL_DATA_DIR = os.path.join(DATA_DIR, 'label_data')
# the benchmark dir
BENCHMARK_DIR = os.path.join(DATA_DIR, "benchmark")
# the doc dir
DOC_DIR = os.path.join(ROOT_DIR, 'doc')