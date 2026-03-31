import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.net_drought_rgb import DroughtClassifierRGBLite
from datasets.dataset_drought import build_dataloaders
print("Modules imported successfully!")
