import os
import urllib.request
import pandas as pd
import numpy as np

TRAIN_URLS = [
    "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv",
    "https://raw.githubusercontent.com/princee1/UNSW_NB15-Data-analysis/master/UNSW_NB15_training-set.csv",
    "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/UNSW_NB15_training-set.csv"
]

TEST_URLS = [
    "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_testing-set.csv",
    "https://raw.githubusercontent.com/princee1/UNSW_NB15-Data-analysis/master/UNSW_NB15_testing-set.csv",
    "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/UNSW_NB15_testing-set.csv"
]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")

def download_file(urls: list, dest_path: str):
    if os.path.exists(dest_path):
        print(f"[INFO] File already exists at {dest_path}. Skipping download.")
        return True

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    for url in urls:
        try:
            print(f"[INFO] Attempting download from {url}...")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"[SUCCESS] Download completed: {dest_path}")
            return True
        except Exception as e:
            print(f"[WARNING] Failed to download from {url}: {e}")
    return False

def generate_fallback_dataset(train_path: str, test_path: str):
    """Generates synthetic UNSW-NB15 dataset matching original schema if network downloads fail."""
    print("[WARNING] Could not fetch raw CSVs via network. Generating synthetic UNSW-NB15 dataset for local development...")
    
    attack_cats = ['Normal', 'Generic', 'Exploits', 'Fuzzers', 'DoS', 'Reconnaissance', 'Analysis', 'Backdoor', 'Shellcode', 'Worms']
    probs = [0.45, 0.20, 0.12, 0.08, 0.06, 0.04, 0.02, 0.01, 0.01, 0.01]
    
    def build_df(num_samples):
        np.random.seed(42)
        cats = np.random.choice(attack_cats, size=num_samples, p=probs)
        labels = (cats != 'Normal').astype(int)
        
        data = {
            'id': np.arange(1, num_samples + 1),
            'dur': np.random.exponential(scale=1.5, size=num_samples),
            'proto': np.random.choice(['tcp', 'udp', 'unas', 'arp', 'ospf'], size=num_samples),
            'service': np.random.choice(['-', 'http', 'dns', 'ftp', 'smtp', 'ssh'], size=num_samples),
            'state': np.random.choice(['FIN', 'CON', 'INT', 'REQ', 'RST'], size=num_samples),
            'spkts': np.random.poisson(lam=10, size=num_samples),
            'dpkts': np.random.poisson(lam=8, size=num_samples),
            'sbytes': np.random.randint(40, 10000, size=num_samples),
            'dbytes': np.random.randint(40, 20000, size=num_samples),
            'rate': np.random.uniform(0.1, 1000.0, size=num_samples),
            'sttl': np.random.choice([31, 62, 254, 255], size=num_samples),
            'dttl': np.random.choice([0, 29, 60, 252], size=num_samples),
            'sload': np.random.uniform(1000, 1000000, size=num_samples),
            'dload': np.random.uniform(1000, 1000000, size=num_samples),
            'sloss': np.random.randint(0, 5, size=num_samples),
            'dloss': np.random.randint(0, 5, size=num_samples),
            'sinpkt': np.random.uniform(0.01, 100.0, size=num_samples),
            'dinpkt': np.random.uniform(0.01, 100.0, size=num_samples),
            'sjit': np.random.uniform(0, 500, size=num_samples),
            'djit': np.random.uniform(0, 500, size=num_samples),
            'swin': np.random.choice([0, 255], size=num_samples),
            'stcpb': np.random.randint(0, 10000000, size=num_samples),
            'dtcpb': np.random.randint(0, 10000000, size=num_samples),
            'dwin': np.random.choice([0, 255], size=num_samples),
            'tcprtt': np.random.uniform(0.001, 0.5, size=num_samples),
            'synack': np.random.uniform(0.001, 0.25, size=num_samples),
            'ackdat': np.random.uniform(0.001, 0.25, size=num_samples),
            'smean': np.random.randint(40, 1000, size=num_samples),
            'dmean': np.random.randint(40, 1000, size=num_samples),
            'trans_depth': np.random.randint(0, 3, size=num_samples),
            'response_body_len': np.random.randint(0, 5000, size=num_samples),
            'ct_srv_src': np.random.randint(1, 20, size=num_samples),
            'ct_state_ttl': np.random.randint(0, 6, size=num_samples),
            'ct_dst_ltm': np.random.randint(1, 20, size=num_samples),
            'ct_src_dport_ltm': np.random.randint(1, 20, size=num_samples),
            'ct_dst_sport_ltm': np.random.randint(1, 20, size=num_samples),
            'ct_dst_src_ltm': np.random.randint(1, 20, size=num_samples),
            'is_ftp_login': np.random.choice([0, 1], size=num_samples),
            'ct_ftp_cmd': np.random.randint(0, 4, size=num_samples),
            'ct_flw_http_mthd': np.random.randint(0, 4, size=num_samples),
            'ct_src_ltm': np.random.randint(1, 20, size=num_samples),
            'ct_srv_dst': np.random.randint(1, 20, size=num_samples),
            'is_sm_ips_ports': np.random.choice([0, 1], size=num_samples),
            'attack_cat': cats,
            'label': labels
        }
        return pd.DataFrame(data)

    df_train = build_df(10000)
    df_test = build_df(2500)
    
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    print(f"[SUCCESS] Synthetic UNSW-NB15 training dataset created at {train_path}")
    print(f"[SUCCESS] Synthetic UNSW-NB15 testing dataset created at {test_path}")

def main():
    train_path = os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv")
    test_path = os.path.join(DATA_DIR, "UNSW_NB15_testing-set.csv")

    s1 = download_file(TRAIN_URLS, train_path)
    s2 = download_file(TEST_URLS, test_path)

    if not (s1 and s2):
        generate_fallback_dataset(train_path, test_path)

if __name__ == "__main__":
    main()


