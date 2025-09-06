from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import os 
load_dotenv() 


@dataclass
class Config:
    mnt_path = Path('/mnt/gis/')
    label_studio_token = os.environ.get('LABEL_STUDIO_API_TOKEN')
    label_studio_url = os.environ.get('LABEL_STUDIO_URL')
    label_studio_project=4