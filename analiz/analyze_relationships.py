import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore', category=FutureWarning)
sns.set_theme(style="whitegrid", font_scale=1.1)

DATA_DIR = "/home/kekec/usta/final_dataset_parquets"
parquet_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
print(f"Found {len(parquet_files)} parquet files.")

dfs = []
for f in parquet_files:
    try:
        df = pd.read_parquet(f)
        df['session_label'] = os.path.basename(f).replace('.parquet', '')
        if 'frame_interaction_valid' in df.columns:
            df = df[df['frame_interaction_valid'] == True]
        dfs.append(df)
    except Exception as e:
        print(f"Error loading {f}: {e}")

if not dfs:
    print("No valid data loaded. Exiting.")
    exit(1)

data = pd.concat(dfs, ignore_index=True)
print(f"Total valid frames across all sessions: {len(data):,}")

WINDOW_SIZE = 60 # e.g. 2 seconds at 30fps
STEP_SIZE   = 30 # 1 second overlap

def compute_window_features(w):
    feat = {}
    
    # 1. Proxemics and Approaching/Withdrawing
    # We look at distance, and crucially, the DELTA (change) in distance over the window.
    # A negative delta means they got closer (approaching). Positive means moving apart (withdrawing).
    if 'dyad_root_distance' in w.columns:
        s = w['dyad_root_distance'].dropna()
        if len(s) > 1:
            feat['dyad_distance_mean'] = s.mean()
            # Calculate velocity of approach (negative = getting closer)
            feat['dyad_distance_delta'] = s.iloc[-1] - s.iloc[0]

    # 2. Reaching vs Pulling Back (Hand to other person's head)
    # If delta is negative, the hand is reaching towards the other person.
    # If delta is positive, the hand is pulling away.
    for person, other in [('p1', 'p2'), ('p2', 'p1')]:
        left_dist_col = f'{person}_left_wrist_to_{other}_head_distance'
        right_dist_col = f'{person}_right_wrist_to_{other}_head_distance'
        
        # We will take the minimum distance of the two hands as "closest hand"
        if left_dist_col in w.columns and right_dist_col in w.columns:
            # Align the two series by dropping rows where *either* is NaN
            valid_idx = w[left_dist_col].notna() & w[right_dist_col].notna()
            left_s = w.loc[valid_idx, left_dist_col]
            right_s = w.loc[valid_idx, right_dist_col]
            
            if len(left_s) > 1 and len(right_s) > 1:
                # Combine hands to find the closest hand at each valid frame
                closest_hand = np.minimum(left_s.values, right_s.values)
                
                feat[f'{person}_closest_hand_to_other_dist'] = closest_hand.mean()
                feat[f'{person}_hand_reach_delta'] = closest_hand[-1] - closest_hand[0]
                
    # 3. Gaze interaction
    for person in ['p1', 'p2']:
        gaze_col = f'{person}_gaze_to_other_head_angle_deg'
        if gaze_col in w.columns:
            s = w[gaze_col].dropna()
            if len(s) > 1:
                feat[f'{person}_gaze_angle_mean'] = s.mean()
                
    # 4. Motion energy
    for person in ['p1', 'p2']:
        mcol = f'{person}_motion_speed'
        if mcol in w.columns:
            s = w[mcol].dropna()
            if len(s) > 1:
                feat[f'{person}_motion_speed'] = s.mean()
                
    return feat

def swap_features(feat):
    """Symmetric swap to ensure clustering is role-agnostic."""
    swap = {}
    for k, v in feat.items():
        if k.startswith('p1_'):
            swap[k.replace('p1_', 'p2_')] = v
        elif k.startswith('p2_'):
            swap[k.replace('p2_', 'p1_')] = v
        else:
            swap[k] = v
    return swap

def extract_windows(data):
    all_windows = []
    sessions = data.groupby('session_label')
    for session_name, session_df in sessions:
        session_df = session_df.sort_values('frame_idx').reset_index(drop=True)
        n = len(session_df)
        
        for start in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
            end = start + WINDOW_SIZE
            w = session_df.iloc[start:end]
            
            feat = compute_window_features(w)
            if not feat:
                continue
                
            feat['session_label'] = session_name
            feat['window_idx'] = start
            
            # Original
            f1 = feat.copy()
            f1['is_swapped'] = 0
            all_windows.append(f1)
            
            # Swapped
            f2 = swap_features(feat)
            f2['session_label'] = session_name
            f2['window_idx'] = start
            f2['is_swapped'] = 1
            all_windows.append(f2)
            
    return pd.DataFrame(all_windows)

print("Extracting features from sliding windows (including deltas for reactions)...")
window_df = extract_windows(data)
print(f"Extracted {len(window_df):,} windows.")

meta_cols = ['session_label', 'window_idx', 'is_swapped']
feature_cols = [c for c in window_df.columns if c not in meta_cols]

# Clean
window_clean = window_df.dropna(subset=feature_cols).copy()
print(f"Windows available for clustering: {len(window_clean):,}")

if len(window_clean) == 0:
    print("Not enough valid windows. Check data.")
    exit(1)

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(window_clean[feature_cols].values)

# Cluster into 5 relationship states
K = 6
km = KMeans(n_clusters=K, random_state=42, n_init=10)
window_clean['cluster'] = km.fit_predict(X)

print("Clustering completed.")
print("Silhouette Score:", silhouette_score(X, window_clean['cluster'].values, sample_size=2000))

# Profile the clusters
profiles = window_clean.groupby('cluster')[feature_cols].mean()

# For easier reading, swap p1/p2 if p2 is moving faster, so "Person A" is always the active one
for idx in profiles.index:
    speed_a = profiles.loc[idx, 'p1_motion_speed']
    speed_b = profiles.loc[idx, 'p2_motion_speed']
    if speed_b > speed_a:
        # Swap all A and B columns
        for c in list(profiles.columns):
            if c.startswith('p1_'):
                b_col = c.replace('p1_', 'p2_')
                temp = profiles.loc[idx, c]
                profiles.loc[idx, c] = profiles.loc[idx, b_col]
                profiles.loc[idx, b_col] = temp

# Normalize between 0 and 1 for heatmap visualization
# Instead of standard min-max, we will do zero-centered max-abs for delta values
# so negative values (approaching) show as cold, positive (withdrawing) as hot.
norm_profiles = profiles.copy()
for col in norm_profiles.columns:
    if "delta" in col:
        max_abs = norm_profiles[col].abs().max()
        if max_abs > 0:
            norm_profiles[col] = norm_profiles[col] / max_abs
    else:
        cmin = norm_profiles[col].min()
        cmax = norm_profiles[col].max()
        if cmax > cmin:
            norm_profiles[col] = (norm_profiles[col] - cmin) / (cmax - cmin)

# Rename columns nicely
def rename_for_display(c):
    c = c.replace('p1_', 'Person_A_').replace('p2_', 'Person_B_')
    return c

norm_profiles.columns = [rename_for_display(c) for c in norm_profiles.columns]

plt.figure(figsize=(16, 10))
# Center colormap at 0 for deltas (which range -1 to 1), while others range 0 to 1
sns.heatmap(norm_profiles.T.sort_index(), annot=True, fmt='.2f', cmap='vlag', center=0, linewidths=0.5)
plt.title('Non-Verbal Relationship States (Including Approach/Withdraw Dynamics)', fontsize=16)
plt.tight_layout()
plt.savefig('relationship_clusters_heatmap.png', dpi=150)
print("Saved heatmap to relationship_clusters_heatmap.png")

# Also save cluster sizes
cluster_counts = window_clean[window_clean["is_swapped"] == 0]['cluster'].value_counts().sort_index()
print("\nCluster Occurrences (original un-swapped windows):")
for cluster_id, count in cluster_counts.items():
    print(f"Cluster {cluster_id}: {count} windows")

# Save a CSV with cluster assignments for the sessions
window_clean[window_clean["is_swapped"] == 0][['session_label', 'window_idx', 'cluster']].to_csv("session_clusters.csv", index=False)
print("Saved session cluster timeline to session_clusters.csv")

