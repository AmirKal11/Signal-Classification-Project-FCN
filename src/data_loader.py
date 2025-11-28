import h5py
import numpy as np
import pandas as pd
import os
import argparse

class H5DataLoader:
    def __init__(self, default_signal_path: str = None, default_background_path: str = None):
        self.default_signal_path = default_signal_path
        self.default_background_path = default_background_path

    def get_data(self, signal_path: str = None, background_path: str = None) -> dict:
        sig_p = signal_path if signal_path else self.default_signal_path
        bkg_p = background_path if background_path else self.default_background_path
        if not sig_p or not bkg_p: raise ValueError("Paths must be provided.")

        print(f"Loading Signal: {sig_p}")
        sig_components, n_sig_events = self._load_components(sig_p, is_signal=True)
        print(f"Loading Background: {bkg_p}")
        bkg_components, _ = self._load_components(bkg_p, is_signal=False, index_offset=n_sig_events)

        print("Combining datasets...")
        full_data = {}
        for key in sig_components.keys():
            full_data[key] = pd.concat([sig_components[key], bkg_components[key]], ignore_index=True)
            
        print(f"Data loaded. Keys: {list(full_data.keys())}")
        return full_data

    def _load_components(self, file_path: str, is_signal: bool, index_offset: int = 0) -> tuple:
        if not os.path.exists(file_path): raise FileNotFoundError(f"File not found: {file_path}")
        components = {}
        with h5py.File(file_path, 'r') as f:
            n_events = f['event'].shape[0] if 'event' in f else f['jet'].shape[0]
            event_ids = np.arange(1, n_events + 1) + index_offset

            components['event'] = self._structured_to_df(f['event'], event_ids, deduplicate=True)
            components['met'] = self._structured_to_df(f['met'], event_ids, deduplicate=True)
            
            components['ditau'] = self._clean_columns(
                self._structured_to_df(f['ditau'], event_ids, deduplicate=False), 'ditau'
            )
            components['jet'] = self._clean_columns(
                self._structured_to_df(f['jet'], event_ids, deduplicate=False), 'jet'
            )
            
            df_elec = self._clean_columns(self._structured_to_df(f['elec'], event_ids), 'elec')
            df_muon = self._clean_columns(self._structured_to_df(f['muon'], event_ids), 'muon')
            components['lepton'] = self._merge_leptons(df_elec, df_muon)

            components['event']['signal'] = 1 if is_signal else 0
        return components, n_events

    def _structured_to_df(self, dataset, event_ids, deduplicate: bool = False) -> pd.DataFrame:
        data_dict = {}
        keys = dataset.dtype.names
        if not keys: return pd.DataFrame()
        sample_col = np.array(dataset[keys[0]]).ravel()
        
        if len(sample_col) == len(event_ids): expanded_ids = event_ids
        else: expanded_ids = np.repeat(event_ids, len(sample_col) // len(event_ids))
            
        data_dict['event_index'] = expanded_ids
        for k in keys: data_dict[k] = np.array(dataset[k]).ravel()
            
        df = pd.DataFrame(data_dict)
        if deduplicate: return df.drop_duplicates(subset='event_index', keep='first').reset_index(drop=True)
        return df

    def _clean_columns(self, df: pd.DataFrame, particle_type: str) -> pd.DataFrame:
        rename_map = {}
        ignore_terms = ['index', 'final'] 
        
        rename_map = {}
        ignore_terms = ['index', 'final'] 
        
        # FIX: Changed 'BDT' to 'bdt', 'Jvt' to 'jvt', etc. to match Preprocessor expectations
        keywords = {
            'pt': 'pt', 'eta': 'eta', 'phi': 'phi', 'mass': 'm', 
            'energy': 'e', 
            'bdt': 'bdt',       # <--- WAS 'BDT', CHANGED TO 'bdt'
            'jvt': 'jvt',       # <--- WAS 'Jvt', CHANGED TO 'jvt'
            'isol': 'iso', 'iso': 'iso', 
            'bjet': 'bjet', 'btag': 'btag_score', # Changed to lowercase just in case
            'm': 'm', 'e': 'e' 
        }
        
        for col in df.columns:
            col_lower = col.lower()
            if particle_type not in col_lower: continue
            if 'sub' in col_lower: continue

            if 'num' in col_lower:
                if 'bjet' in col_lower or 'btag' in col_lower: rename_map[col] = 'num_bjet'
                else: rename_map[col] = f"num_{particle_type}"
                continue

            if any(term in col_lower for term in ignore_terms): continue

            
            suffix_check = col_lower.replace(particle_type, "", 1)

            for key, suffix in keywords.items():
                if key in suffix_check:
                    rename_map[col] = f"{particle_type}_{suffix}"
                    break
        return df.rename(columns=rename_map)

    def _merge_leptons(self, df_e, df_m) -> pd.DataFrame:
        df_lep = pd.DataFrame()
        df_lep['event_index'] = df_e['event_index']
        e_pt = 'elec_pt' if 'elec_pt' in df_e.columns else None
        m_pt = 'muon_pt' if 'muon_pt' in df_m.columns else None
        
        if not e_pt: df_e['elec_pt'] = 0.0
        if not m_pt: df_m['muon_pt'] = 0.0

        has_elec = df_e['elec_pt'] > 0
        for var in ['pt', 'eta', 'phi', 'iso']:
            e_col = f"elec_{var}"
            m_col = f"muon_{var}"
            target = f"lepton_{var}"
            val_e = df_e[e_col] if e_col in df_e.columns else 0.0
            val_m = df_m[m_col] if m_col in df_m.columns else 0.0
            df_lep[target] = np.where(has_elec, val_e, val_m)
        return df_lep