import numpy as np
import pandas as pd
import pickle
from pathlib import Path

def add_trial_columns(
        trial_info: pd.DataFrame
) -> pd.DataFrame:
    
    # ----
    # Add a column for whether previous trial was a violation 
    violation = trial_info["violated"].values
    prev_violation = np.roll(violation, 1)
    prev_violation[0] = np.nan   # First trial has no previous trial
    trial_info["prev_violated"] = prev_violation


    # ----
    # Add a column for whether next trial was a violation 
    violation = trial_info["violated"].values
    next_violation = np.roll(violation, -1)
    next_violation[1] = np.nan   # Last trial has no next trial
    trial_info["next_violated"] = next_violation

    # ----
    # Now pick out only trials that were not violation trials, and in which poked_r is not None,
    # meaning there was a side poke:
    trial_info = trial_info[(trial_info["violated"]==False) & (trial_info["poked_r"].notnull())]
    trial_info = trial_info.copy()  # for some reason this prevents a slicing error below
    nTrials = len(trial_info)

    # ----
    # Add a column for whether previous trial was right choice 
    poked_r = trial_info["poked_r"].values
    prev_poked_r = np.roll(poked_r, 1)
    prev_poked_r[0] = np.nan # First trial has no previous trial
    trial_info["prev_poked_r"] = prev_poked_r

    # ----
    # Add a column for whether previous previous trial was right choice 
    poked_r = trial_info["poked_r"].values
    prev_prev_poked_r = np.roll(poked_r, 2)
    prev_prev_poked_r[0] = np.nan # First  trial has no previous previous trial
    prev_prev_poked_r[1] = np.nan # Second trial has no previous previous trial
    trial_info["prev_prev_poked_r"] = prev_prev_poked_r

    # ----
    # Add a column for whether next trial is right choice
    poked_r = trial_info["poked_r"].values
    next_poked_r = np.roll(poked_r, -1)
    next_poked_r[-1] = np.nan # Last trial has no next trial
    trial_info["next_poked_r"] = next_poked_r

    # ----
    # Add a column for whether previous trial had is_hit = 1
    is_hit = trial_info["is_hit"].values
    prev_is_hit = np.roll(is_hit, 1)
    prev_is_hit[0] = np.nan # First trial has no previous trial
    trial_info["prev_is_hit"] = prev_is_hit

    # ----
    # Add a column for whether next trial has is_hit = 1
    is_hit = trial_info["is_hit"].values
    next_is_hit = np.roll(is_hit, -1)
    next_is_hit[-1] = np.nan # Last trial has no next trial
    trial_info["next_is_hit"] = next_is_hit

    return trial_info


def add_hemisphere(
        cell_info: pd.DataFrame,
        probe_info: pd.DataFrame
) -> pd.DataFrame:

    # Broadcast type so that we can merge
    cell_info['probe_serial'] = cell_info['probe_serial'].astype(str)
    probe_info['serial'] = probe_info['serial'].astype(str)


    # Merge to match probe serial number
    cell_info = cell_info.merge(
        probe_info,
        left_on='probe_serial',
        right_on='serial',
        how='left',
    )

    # Indicate hemisphere on brain region
    cell_info['brain_region'] = cell_info.apply(
        lambda row: f'{row["hemisphere"]}_{row["brain_region"]}' if pd.notna(row["hemisphere"]) else row["brain_region"],
        axis=1
    )

    # Remove duplicated serial column
    cell_info = cell_info.drop(columns=probe_info.columns)

    return cell_info 



def load_session(
        root: Path,
        ratname: str,
        sessdate: str
) -> tuple[pd.DataFrame]:
    
    file_path = root / ratname / Path(f'{ratname}_{sessdate}_CellsAndTrials.pkl')
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        dfTrials = data['dfTrials']
        dfCells = data['dfCells']
        dfCells.reset_index(inplace=True)
    
    dfTrials = add_trial_columns(dfTrials)

    return dfTrials, dfCells


def dump_useless_cells(
        cell_info: pd.DataFrame
) -> pd.DataFrame:
    
    brain_region = np.setdiff1d(
        ar1 = cell_info['brain_region'].unique(),
        ar2 = np.array(["CC", "unknown"])
    )
    print("We have %d brain regions: %s" % (len(brain_region), brain_region))
    cell_info = cell_info[cell_info['brain_region'].isin(brain_region)]

    spike_times = cell_info['raw_spike_time_s'].values
    n_cells = len(spike_times)
    print("We have %d cells total" % n_cells)

    return cell_info


def assign_brain_regions(dfCells, regions_df):
    """
    Assign brain regions to neurons based on probe serial and electrode number.
    
    Parameters
    ----------
    dfCells : pd.DataFrame
        Neuron dataframe with 'probe_serial' and 'electrode' columns
    regions_df : pd.DataFrame
        Brain regions dataframe with 'serial', 'region', 'min_electrode', 'max_electrode'
    
    Returns
    -------
    pd.Series
        Brain region for each neuron
    """
    # Convert serial to string for matching
    regions_df = regions_df.copy()
    regions_df['serial'] = regions_df['serial'].astype(str)
    
    # Handle 'Inf' in max_electrode
    regions_df['max_electrode'] = pd.to_numeric(regions_df['max_electrode'], errors='coerce')
    regions_df.loc[regions_df['max_electrode'].isna(), 'max_electrode'] = np.inf
    
    brain_regions = []
    
    for _, neuron in dfCells.iterrows():
        serial = str(neuron['probe_serial'])
        electrode = neuron['electrode']
        
        # Find matching probe regions
        probe_regions = regions_df[regions_df['serial'] == serial]
        
        # Find the region where electrode falls in range
        match = probe_regions[
            (probe_regions['min_electrode'] <= electrode) & 
            (electrode <= probe_regions['max_electrode'])
        ]
        
        if len(match) == 1:
            brain_regions.append(match['region'].values[0])
        elif len(match) > 1:
            # Multiple matches (could be multi-shank probe) - take first
            brain_regions.append(match['region'].values[0])
        else:
            brain_regions.append('unknown')
    
    return pd.Series(brain_regions, index=dfCells.index)