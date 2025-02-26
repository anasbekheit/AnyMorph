import os

HOME = os.environ['HOME']
SRC_DIR = f"{HOME}/amorpheus/modular-rl/src"
ENV_DIR = f"{SRC_DIR}/environments"
XML_DIR = f"{ENV_DIR}/xmls"
BASE_MODULAR_ENV_PATH = f"{ENV_DIR}/ModularEnv.py"
BASELINES_DIR = f"{HOME}/store/nosnap/morphology-baselines"
DATA_DIR = f"{BASELINES_DIR}/results"
BUFFER_DIR = f"{BASELINES_DIR}/buffers"
VIDEO_DIR = f"{BASELINES_DIR}/videos"
VIDEO_RESOLUATION = (480, 480)