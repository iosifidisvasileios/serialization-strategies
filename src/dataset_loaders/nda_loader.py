from __future__ import annotations
from .base_loader import BaseDatasetLoader
class NDALoader(BaseDatasetLoader):
    dataset_name = 'ndas'
    default_label_map = {'Party':'Party', 'Effective Date':'Effective Date', 'Jurisdiction':'Jurisdiction'}
