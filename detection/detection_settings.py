import os
from settings import proj_dir
cur_root = os.path.dirname(__file__)
MODEL_CKPT_PATH = os.path.join(cur_root, 'model')
DEMO_OUTPUT_DIR = os.path.join(cur_root, 'data', 'res')
DEMO_INPUT_DIR = os.path.join(cur_root, 'data', 'demo')

MODEL_EXPORT_PATH = os.path.join(proj_dir, 'exported_model', 'detection')
