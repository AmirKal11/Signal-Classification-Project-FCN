import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# CONSTANTS (MeV)
LEPTON_PT_CUT = 27000
LEPTON_ETA_CUT = 2.47
JET_PT_CUT = 20000
JET_ETA_CUT = 2.8
DITAU_PT_CUT = 50000
DITAU_ETA_CUT = 2.5
DITAU_BDT_CUT = 0.02

class Logic_PhysicsPreprocessor:
    def __init__(self, data_dict):
        self.data = data_dict
        self.events = None 
        self.scaler = StandardScaler()
        
        # EXACT 33 FEATURES FROM TABLE 3
        self.feature_columns = [
            # 1. Event & MET (5 variables)
            'averageInteractionsPerCrossing', 
            'MetTST_met', 'MetTST_sumet', 'MetTrack_met', 'MetTrack_sumet',
            
            # 2. Counts (3 variables)
            'num_jet', 'num_bjet', 'num_ditau',

            # 3. Lepton (3 variables - 1 particle)
            'lepton_pt_1', 'lepton_eta_1', 'lepton_phi_1',

            # 4. DiTau (10 variables - 1 particle)
            # Added 'ditau_phi_1' to match the 33 variable count
            'ditau_pt_1', 'ditau_eta_1', 'ditau_phi_1', 'ditau_bdt_1',
            # Note: Paper lists these subjet vars. 
            # Ensure your data loader names them exactly like this or similar.
            'ditau_subjet_lead_pt_1', 'ditau_subjet_lead_eta_1', 'ditau_subjet_lead_phi_1',
            'ditau_subjet_subl_pt_1', 'ditau_subjet_subl_eta_1', 'ditau_subjet_subl_phi_1',

            # 5. Jets (8 variables - Top 2 particles)
            'jet_pt_1', 'jet_eta_1', 'jet_phi_1', 'jet_jvt_1',
            'jet_pt_2', 'jet_eta_2', 'jet_phi_2', 'jet_jvt_2',

            # 6. B-Jet (4 variables - Top 1 particle)
            'bjet_pt_1', 'bjet_eta_1', 'bjet_phi_1', 'bjet_jvt_1'
        ]

    def _calculate_counts(self):
        """Calculates multiplicities using the Ground Truth from the file where possible"""
        
        # --- 1. JETS ---
        if 'num_jet' in self.data['jet'].columns:
            n_jets = self.data['jet'].groupby('event_index')['num_jet'].max().reset_index(name='num_jet')
        else:
            jets = self.data['jet']
            valid_jets = jets[(jets['jet_pt'] > JET_PT_CUT) & (jets['jet_eta'].abs() < JET_ETA_CUT)]
            n_jets = valid_jets.groupby('event_index').size().reset_index(name='num_jet')

        # --- 2. B-JETS ---
        if 'num_bjet' in self.data['jet'].columns:
            n_bjets = self.data['jet'].groupby('event_index')['num_bjet'].max().reset_index(name='num_bjet')
        else:
            jets = self.data['jet']
            valid_bjets = jets[(jets['jet_bjet'] == 1) & (jets['jet_pt'] > JET_PT_CUT)]
            n_bjets = valid_bjets.groupby('event_index').size().reset_index(name='num_bjet')

        # --- 3. DITAUS ---
        taus = self.data['ditau']
        mask_tau = (taus['ditau_pt'] > DITAU_PT_CUT) & (taus['ditau_eta'].abs() < DITAU_ETA_CUT)
        if 'ditau_bdt' in taus.columns:
            mask_tau &= (taus['ditau_bdt'] > DITAU_BDT_CUT)
        
        valid_taus = taus[mask_tau]
        n_taus = valid_taus.groupby('event_index').size().reset_index(name='num_ditau')

        # --- MERGE ---
        self.counts_df = n_jets.merge(n_bjets, on='event_index', how='outer') \
                               .merge(n_taus, on='event_index', how='outer').fillna(0)

    def _get_valid_event_indices(self):
        """Applies the Cut Flow Logic from the Paper"""
        counts = self.counts_df
        
        # 1. Lepton Cut (Usually Pre-filtered, but ensuring >= 1)
        all_indices = counts['event_index'].unique()
        
        # 2. Jet Cut (>= 3 Jets)
        passed_jet = counts[counts['num_jet'] >= 3]['event_index']
        
        # 3. B-Jet Cut (>= 1 B-Jet)
        passed_bjet = counts[counts['num_bjet'] >= 1]['event_index']
        
        # 4. DiTau Cut (>= 1 DiTau)
        passed_tau = counts[counts['num_ditau'] >= 1]['event_index']
        
        # Intersection
        valid_indices = np.intersect1d(passed_jet, passed_bjet)
        valid_indices = np.intersect1d(valid_indices, passed_tau)
        
        print(f"Events passing cuts: {len(valid_indices)}")
        return list(valid_indices)

    def _sort_and_filter(self, df, sort_col, n_keep):
        """Sorts particles by pT and keeps top N, then pivots to wide format"""
        # Sort by Event and then by pT (descending)
        df_sorted = df.sort_values(by=['event_index', sort_col], ascending=[True, False])
        
        # Keep top N
        df_top = df_sorted.groupby('event_index').head(n_keep).copy()
        
        # Create rank (1, 2, ...)
        df_top['rank'] = df_top.groupby('event_index').cumcount() + 1
        

        df_pivot = df_top.pivot(index='event_index', columns='rank')
        
        df_pivot.columns = [f"{col[0]}_{col[1]}" for col in df_pivot.columns]
        
        return df_pivot.reset_index()

    def process_pipeline(self):
        self._calculate_counts()
        valid_indices = self._get_valid_event_indices()
        
        if not valid_indices:
            raise ValueError("No events passed the cuts! Check cuts or input data.")

        # --- 1. Process Each Particle Type ---
        # Jets: Keep Top 2
        # Columns to keep for pivot
        jet_cols = ['event_index', 'jet_pt', 'jet_eta', 'jet_phi', 'jet_jvt']
        jets = self.data['jet'][self.data['jet']['event_index'].isin(valid_indices)]
        # Filter columns if they exist
        jets = jets[[c for c in jet_cols if c in jets.columns]]
        df_jets = self._sort_and_filter(jets, 'jet_pt', 2)
        
        # B-Jets: Keep Top 1
        # Filter for actual b-jets first
        bjets_all = self.data['jet'][(self.data['jet']['jet_bjet'] == 1) & (self.data['jet']['event_index'].isin(valid_indices))]
        # Rename columns to avoid collision with regular jets (jet_pt -> bjet_pt)
        bjets = bjets_all.rename(columns=lambda x: x.replace('jet_', 'bjet_') if 'jet_' in x else x)
        # Select strict columns for B-Jet
        bjet_cols = ['event_index', 'bjet_pt', 'bjet_eta', 'bjet_phi', 'bjet_jvt']
        bjets = bjets[[c for c in bjet_cols if c in bjets.columns]]
        df_bjets = self._sort_and_filter(bjets, 'bjet_pt', 1)
        
        # Leptons: Keep Top 1
        # Check if 'lepton' key exists, else fallback
        lep_source = 'lepton' if 'lepton' in self.data else 'elec' # simplified fallback
        leps = self.data[lep_source][self.data[lep_source]['event_index'].isin(valid_indices)]
        
        # Rename if using raw 'elec'
        if lep_source != 'lepton':
             leps = leps.rename(columns=lambda x: x.replace(f'{lep_source}_', 'lepton_'))
             
        # Strict Column Selection
        lep_cols = ['event_index', 'lepton_pt', 'lepton_eta', 'lepton_phi']
        leps = leps[[c for c in lep_cols if c in leps.columns]]
        df_leps = self._sort_and_filter(leps, 'lepton_pt', 1)

        # DiTaus: Keep Top 1
        taus = self.data['ditau'][self.data['ditau']['event_index'].isin(valid_indices)]
        # Strict Column Selection
        tau_cols = [
            'event_index', 'ditau_pt', 'ditau_eta', 'ditau_phi', 'ditau_bdt',
            'ditau_subjet_lead_pt', 'ditau_subjet_lead_eta', 'ditau_subjet_lead_phi',
            'ditau_subjet_subl_pt', 'ditau_subjet_subl_eta', 'ditau_subjet_subl_phi'
        ]
        taus = taus[[c for c in tau_cols if c in taus.columns]]
        df_taus = self._sort_and_filter(taus, 'ditau_pt', 1)
        
        # --- 2. Merge All ---
        # Start with Event/MET info
        # Strict selection for Event
        ev_cols = ['event_index', 'averageInteractionsPerCrossing', 'signal']
        avail_ev_cols = [c for c in ev_cols if c in self.data['event'].columns]
        if 'event_index' not in avail_ev_cols: avail_ev_cols.append('event_index')
        
        final_df = self.data['event'][self.data['event']['event_index'].isin(valid_indices)][avail_ev_cols].copy()
        
        # Merge MET
        final_df = final_df.merge(self.data['met'], on='event_index', how='inner')
        
        # Merge Counts
        final_df = final_df.merge(self.counts_df, on='event_index', how='inner')
        
        # Merge Particles
        final_df = final_df.merge(df_jets, on='event_index', how='left')
        final_df = final_df.merge(df_bjets, on='event_index', how='left')
        final_df = final_df.merge(df_leps, on='event_index', how='left')
        final_df = final_df.merge(df_taus, on='event_index', how='left')
        
        if 'signal' not in final_df.columns:
            final_df['signal'] = 0 
            
        self.events = final_df.fillna(0)
        
        # DEBUG PRINT
        print(f"Processed Columns ({len(self.events.columns)}): {list(self.events.columns)}")
        return self.events

    def get_train_test_data(self):
        # FIX: Ensure we have the label
        if 'signal' not in self.events.columns:
             raise ValueError("Target column 'signal' missing.")
             
        y = self.events['signal']
        
        missing_cols = [c for c in self.feature_columns if c not in self.events.columns]
        if missing_cols:
            print(f"WARNING: Missing columns: {missing_cols}")
            for c in missing_cols: self.events[c] = 0.0
        
        dead_columns = []
        for col in self.feature_columns:
            # Check if the column is entirely zeros (or close enough to zero)
            # using np.allclose handles float precision issues
            if np.allclose(self.events[col].values, 0, atol=1e-7):
                dead_columns.append(col)
        
        if dead_columns:
            # STOP EVERYTHING and throw a loud error
            raise ValueError(
                f"\n{'='*60}\n"
                f"DATA INTEGRITY ERROR: DEAD COLUMNS DETECTED \n"
                f"{'='*60}\n"
                f"The following feature columns contain ONLY ZEROS:\n"
                f" -> {dead_columns}\n\n"
                f"DIAGNOSIS:\n"
                f"{'='*60}"
            )
            
        X = self.events[self.feature_columns]
        
        print(f"Final Feature Shape: {X.shape}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
        
        # Scale
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)