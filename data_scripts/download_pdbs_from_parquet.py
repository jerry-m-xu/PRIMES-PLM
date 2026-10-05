#!/usr/bin/env python3
import os
import sys
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pandas as pd
import pyarrow.parquet as pq

# === GLOBAL VARIABLES: edit these before running ===
INPUT_PARQUET = "/ocean/projects/bio250072p/jxu23/swiss_under_1000_320M.parquet"  # path to your parquet file
OUTPUT_DIR = "/ocean/projects/bio250072p/jxu23/pdbs"      # Local directory to save PDB files
START_ROW = 0                                      # Row to start processing from (0-based)
MAX_PAIRS = 4000000                                # rows to scan; distinct proteins saturate near 400k well before this
COL1 = "chain_1"                                 # name of first column in parquet
COL2 = "chain_2"                                 # name of second column in parquet

# Download settings - OPTIMIZED FOR SPEED
DOWNLOAD_DELAY = 1                               # delay between downloads (seconds)
TIMEOUT = 30                                     # timeout for requests
MAX_WORKERS = 4                                  # parallel downloads
CHUNK_SIZE = 8192                                # chunk size for downloading files
MAX_RETRIES = 3                                  # Try up to 4 times total
RETRY_DELAY = 5.0                                # Base delay for exponential backoff
# ==================================================

# Try to import requests, otherwise use urllib
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    HAS_REQUESTS = True
except ImportError:
    import urllib.request
    import urllib.error
    HAS_REQUESTS = False

# Thread-safe counters
class ThreadSafeCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            self._value += 1
            return self._value

    @property
    def value(self):
        with self._lock:
            return self._value

def create_session():
    """Create an optimized requests session with connection pooling"""
    if not HAS_REQUESTS:
        return None
    
    session = requests.Session()

    # Configure retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.1,
        status_forcelist=[429, 500, 502, 503, 504],
    )

    # Configure adapter with connection pooling
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=MAX_WORKERS,
        pool_maxsize=MAX_WORKERS
    )

    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session

def download_pdb_from_swissmodel(uniprot_id: str, session=None, timeout: int = 30) -> bool:
    """
    Download PDB file from Swiss-Model REST API and save to local directory.
    Returns True if successful, False otherwise.
    """
    url = f"https://swissmodel.expasy.org/repository/uniprot/{uniprot_id}.pdb?provider=swissmodel"
    local_file_path = os.path.join(OUTPUT_DIR, f"{uniprot_id}.pdb")

    # Check if file already exists locally
    if os.path.exists(local_file_path):
        return True

    for attempt in range(MAX_RETRIES + 1):
        try:
            if HAS_REQUESTS and session:
                resp = session.get(url, timeout=timeout, stream=True)
                if resp.status_code == 200:
                    # Save to local file
                    with open(local_file_path, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                            f.write(chunk)
                    return True
                elif resp.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    print(f"Rate limited (429) for {uniprot_id}, waiting {wait_time:.1f}s before retry {attempt + 1}/{MAX_RETRIES + 1}")
                    time.sleep(wait_time)
                    continue
                elif resp.status_code == 404:
                    print(f"PDB not found (404) for {uniprot_id}")
                    return False
                else:
                    print(f"Warning: HTTP {resp.status_code} for ID {uniprot_id}", file=sys.stderr)
                    return False
            else:
                # Fallback to urllib
                req = urllib.request.Request(url)
                try:
                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        if resp.status == 200:
                            content = resp.read()
                            with open(local_file_path, 'wb') as f:
                                f.write(content)
                            return True
                        elif resp.status == 429:
                            # Rate limited - wait and retry
                            wait_time = RETRY_DELAY * (2 ** attempt)
                            print(f"Rate limited (429) for {uniprot_id}, waiting {wait_time:.1f}s before retry {attempt + 1}/{MAX_RETRIES + 1}")
                            time.sleep(wait_time)
                            continue
                        elif resp.status == 404:
                            print(f"PDB not found (404) for {uniprot_id}")
                            return False
                        else:
                            print(f"Warning: HTTP {resp.status} for ID {uniprot_id}", file=sys.stderr)
                            return False
                except urllib.error.HTTPError as e:
                    if e.code == 429:
                        # Rate limited - wait and retry
                        wait_time = RETRY_DELAY * (2 ** attempt)
                        print(f"Rate limited (429) for {uniprot_id}, waiting {wait_time:.1f}s before retry {attempt + 1}/{MAX_RETRIES + 1}")
                        time.sleep(wait_time)
                        continue
                    elif e.code == 404:
                        print(f"PDB not found (404) for {uniprot_id}")
                        return False
                    else:
                        print(f"Warning: HTTPError {e.code} for ID {uniprot_id}", file=sys.stderr)
                        return False
                except urllib.error.URLError as e:
                    print(f"Warning: URLError {e.reason} for ID {uniprot_id}", file=sys.stderr)
                    return False
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait_time = RETRY_DELAY * (2 ** attempt)
                print(f"Exception for {uniprot_id}, waiting {wait_time:.1f}s before retry {attempt + 1}/{MAX_RETRIES + 1}: {e}")
                time.sleep(wait_time)
                continue
            else:
                print(f"Warning: Exception downloading {uniprot_id} after {MAX_RETRIES + 1} attempts: {e}", file=sys.stderr)
                return False

    return False

def download_worker(args):
    """Worker function for parallel downloads"""
    uniprot_id, session, counter, total_count = args
    idx = counter.increment()
    thread_id = threading.get_ident()  # Get unique thread ID

    print(f"[Thread {thread_id}] [{idx}/{total_count}] Downloading {uniprot_id}...", end=' ', flush=True)

    start_time = time.perf_counter()
    success = download_pdb_from_swissmodel(uniprot_id, session)
    end_time = time.perf_counter()

    if success:
        print(f"OK ({end_time - start_time:.2f}s)")
        return True
    else:
        print("Failed")
        return False

def collect_ids_from_parquet(parquet_path: str, col1: str, col2: str, max_pairs: int, start_row: int = 0) -> "tuple[set, int, int]":
    """
    Collect the unique IDs in columns col1 and col2 over parquet rows
    [start_row, start_row + max_pairs).

    Streams Arrow batches of just the two columns. The source file has
    67M-row row groups, so reading one whole with every column into pandas
    (the previous approach) needed tens of GB and was OOM-killed on a
    shared node.

    Returns:
        tuple: (ids: set, pairs_processed: int, last_row: int)
    """
    ids = set()
    pairs_processed = 0
    last_row = start_row - 1
    batch_size = 65536

    parquet_file = pq.ParquetFile(parquet_path)
    total_rows = parquet_file.metadata.num_rows
    if start_row < 0:
        start_row = 0
    if start_row >= total_rows:
        raise ValueError(f"start_row {start_row} is beyond the number of rows ({total_rows}) in the parquet file")
    end_row = min(start_row + max_pairs, total_rows)

    schema_names = set(parquet_file.schema_arrow.names)
    if col1 not in schema_names or col2 not in schema_names:
        raise ValueError(f"Parquet file must contain columns '{col1}' and '{col2}'")

    group_start = 0
    for group_idx in range(parquet_file.num_row_groups):
        group_rows = parquet_file.metadata.row_group(group_idx).num_rows
        group_end = group_start + group_rows
        if group_end <= start_row:
            group_start = group_end
            continue
        if group_start >= end_row:
            break

        offset = group_start
        for batch in parquet_file.iter_batches(
            batch_size=batch_size, row_groups=[group_idx], columns=[col1, col2]
        ):
            batch_end = offset + batch.num_rows
            if batch_end <= start_row:
                offset = batch_end
                continue
            if offset >= end_row:
                break

            # trim the batch to the window, then work on plain Python lists
            lo = max(start_row - offset, 0)
            hi = min(end_row - offset, batch.num_rows)
            values1 = batch.column(col1).slice(lo, hi - lo).to_pylist()
            values2 = batch.column(col2).slice(lo, hi - lo).to_pylist()
            for val in values1 + values2:
                val = str(val).strip() if val is not None else ""
                if val and val != "nan":
                    ids.add(val)

            pairs_processed += hi - lo
            last_row = offset + hi - 1
            if pairs_processed % (batch_size * 8) < (hi - lo):
                print(f"Scanned {pairs_processed:,} rows, {len(ids):,} unique IDs so far", flush=True)
            offset = batch_end

        group_start = group_end
        if last_row + 1 >= end_row:
            break

    return ids, pairs_processed, last_row

def format_time(seconds):
    """Format seconds into human readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"

def main():
    start_time = time.time()

    print(f"Input Parquet: {INPUT_PARQUET}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Download settings: {MAX_WORKERS} workers, {DOWNLOAD_DELAY}s delay, {TIMEOUT}s timeout")
    print(f"Starting at row: {START_ROW}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect IDs from parquet file
    try:
        ids, pairs_processed, last_row = collect_ids_from_parquet(INPUT_PARQUET, COL1, COL2, MAX_PAIRS, START_ROW)
    except Exception as e:
        print(f"Error while reading parquet file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Processed {pairs_processed} pairs (limit {MAX_PAIRS})")
    print(f"Last row processed: {last_row}")
    print(f"Collected {len(ids)} unique IDs")
    print(f"Downloading PDBs from Swiss-Model to: {OUTPUT_DIR}")
    print("-" * 60)

    # Check existing files in local directory
    existing_files = set()
    if os.path.exists(OUTPUT_DIR):
        for filename in os.listdir(OUTPUT_DIR):
            if filename.endswith('.pdb'):
                # Extract protein ID from filename
                protein_id = filename.replace(".pdb", "")
                existing_files.add(protein_id)

    ids_to_download = [id for id in sorted(ids) if id not in existing_files]
    skip_count = len(ids) - len(ids_to_download)

    if skip_count > 0:
        print(f"⏭️  Skipping {skip_count} already existing files in local directory")

    if not ids_to_download:
        print("All files already exist in local directory!")
        return

    print(f"🚀 Starting parallel download of {len(ids_to_download)} files with {MAX_WORKERS} workers...")

    # Create optimized session
    session = create_session() if HAS_REQUESTS else None

    # Setup counters
    counter = ThreadSafeCounter()
    success_count = 0
    fail_count = 0

    # Download files in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Prepare arguments for workers
        args_list = [(id, session, counter, len(ids_to_download))
                    for id in ids_to_download]

        # Submit all tasks
        future_to_id = {executor.submit(download_worker, args): args[0]
                       for args in args_list}

        # Process completed tasks
        for future in as_completed(future_to_id):
            if future.result():
                success_count += 1
            else:
                fail_count += 1

    # Final summary
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Download Complete!")
    print(f"Success: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Failed: {fail_count}")
    print(f"Last row processed: {last_row}")
    print(f"To resume, set START_ROW = {last_row + 1}")
    print(f"Files saved to: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Total time: {format_time(total_time)}")
    if len(ids_to_download) > 0:
        print(f"Average time per file: {total_time/len(ids_to_download):.2f}s")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()

