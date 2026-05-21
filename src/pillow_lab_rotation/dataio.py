import numpy as np
from scipy.io import loadmat

import jax
import jax.numpy as jnp

import os
from pathlib import Path


DATA_ROOT = Path('/Users/jeremyschroeter/Desktop/the-ark/phd/rotations/pillow_lab_rotation/data/runyan_data')


class RunyanSession:

    def __init__(
            self,
            mouse: str,
            date: str
    ):
        self.mouse = mouse
        self.date = date
        self.session_path = DATA_ROOT / mouse / date

        self._load_mats()

        self.dff = self.activity.dff
        self.cellids = self.clustering.cellids
        self.fs = float(self.activity.Fall.ops.fs)

        self.n_e = int((self.cellids == 0).sum())
        self.n_som = int((self.cellids == 1).sum())
        self.n_pv = int((self.cellids == 2).sum())

        self.n_som_evt = sum(ev.condition == 'SOM' for ev in self.D_onsets)
        self.n_pv_evt  = sum(ev.condition == 'PV'  for ev in self.D_onsets)

    
    def _load_mats(self):

        self.activity = loadmat(
            self.session_path / 'activity.mat',
            squeeze_me=True,
            struct_as_record=False
        )['combined_info']

        self.clustering = loadmat(
            self.session_path / 'clustering.mat',
            squeeze_me=True,
            struct_as_record=False
        )['clustering_info']
        
        self.D_onsets = loadmat(
            self.session_path / 'D_onsets_v2.mat',
            squeeze_me=True,
            struct_as_record=False
        )['D_onsets']

    
    def build_event_aligned_tensor
    

