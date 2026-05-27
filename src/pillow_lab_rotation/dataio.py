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

        self._preproc()
        self._build_event_aligned_trial_tensor()

        self.is_som = self.conditions == 'SOM'
        self.is_pv = self.conditions == 'PV'

        self.Ne = len(self.e_idx)
        self.Ni = len(self.i_idx)
        self.N = self.Ne + self.Ni

    
    def _load_mats(self):
        '''
        Load matlab files
        '''

        activity = loadmat(
            self.session_path / 'activity.mat',
            squeeze_me=True,
            struct_as_record=False
        )
        self.activity = activity['combined_info']

        clustering = loadmat(
            self.session_path / 'clustering.mat',
            squeeze_me=True,
            struct_as_record=False
        )
        self.clustering = clustering['clustering_info']
        
        D_onsets = loadmat(
            self.session_path / 'D_onsets_v2.mat',
            squeeze_me=True,
            struct_as_record=False
        )
        self.D_onsets = D_onsets['D_onsets']


    
    def _preproc(self):
        '''
        1. Reorder neurons in excitation and inhibitory rows
        2. Mean subtraction
        '''
        self.e_idx = np.where(self.cellids == 0)[0]
        self.i_idx = np.where((self.cellids == 1) | (self.cellids == 2))[0]
        self.ei_order = np.concatenate([self.e_idx, self.i_idx])

        self.Y_raw = self.dff[self.ei_order]
        self.Y_demeaned = self.Y_raw - self.Y_raw.mean(axis=1, keepdims=True)


    def _build_event_aligned_trial_tensor(self):
        '''
        1. Break into trials
        2. Reshape to work with CTDS implementation (n_trials, T, N, 1)
        '''

        self.pre_window = 101
        self.window = self.D_onsets[0].dff.shape[1]

        self.onsets = np.array([ev.onset for ev in self.D_onsets])
        self.conditions = np.array([ev.condition for ev in self.D_onsets])

        self.windows = np.stack([
            self.Y_demeaned[:, o - self.pre_window : o - self.pre_window + self.window]
            for o in self.onsets
        ])

        self.obs_evt = self.windows.transpose(0, 2, 1)[..., None].astype(np.float32)



