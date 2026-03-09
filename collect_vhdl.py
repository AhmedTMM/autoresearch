"""
Collect VHDL source code from GitHub into parquet shards for autoresearch.

Usage:
    python collect_vhdl.py                  # full collection (~1000 repos)
    python collect_vhdl.py --max-repos 10   # limit repos (for testing)
    python collect_vhdl.py --skip-clone     # only process already-cloned repos

Sources:
    - GitHub search (top VHDL repos by stars)
    - GitHub search (VHDL repos by recent activity)
    - GitHub code search (files with VHDL extensions)
    - Well-known VHDL project lists (OpenCores mirrors, FPGA projects, etc.)

Writes parquet shards to ~/.cache/autoresearch/data/
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
CLONE_DIR = os.path.join(CACHE_DIR, "vhdl_repos")
SHARD_SIZE = 500  # documents per shard
MIN_LINES = 10
MAX_FILE_SIZE = 500_000  # skip files > 500KB (likely generated)
VHDL_EXTENSIONS = (".vhd", ".vhdl", ".vho")

# Well-known VHDL-heavy repos and OpenCores mirrors on GitHub
KNOWN_VHDL_REPOS = [
    "VLSI-EDA/PoC",
    "FPGAwars/FPGA-peripherals",
    "hamsjeong/VHDL",
    "ricardoseriani/VHDL-Pong",
    "dbhi/vhdl-cfg",
    "INTI-CMNB/FPGA_lib",
    "pConst/basic_vhdl",
    "tmeissner/cryptocores",
    "xesscorp/VHDL_Lib",
    "myriadrf/LimeSDR-USB_GW",
    "OSVVM/OSVVM",
    "OSVVM/OsvvmLibraries",
    "VUnit/vunit",
    "ghdl/ghdl",
    "antonblanchard/microwatt",
    "stnolting/neorv32",
    "ZipCPU/zipcpu",
    "openhwgroup/cva6",
    "SpinalHDL/VhdlParser",
    "amb5l/tyto2",
    "MJoergen/MEGA65",
    "mfro/FPGA_vhdl",
    "cassuto/morphern",
    "wltr/common-vhdl",
    "Paebbels/JSON-for-VHDL",
    "FrankBuss/FPGA",
    "progranism/Open-Source-FPGA-Bitcoin-Miner",
    "hoglet67/AtomFpga",
    "emb-lib/fixed_point_math",
    "myriadrf/LimeSDR-Mini_GW",
    "analogdevicesinc/hdl",
    "MUSIC-IN-MY-DIARY/FPGA-Zynq",
    "suoto/fpga_cores",
    "tcjeong/VHDL-labs",
    "NJDFan/register-map",
    "Efinix-Inc/sapphire-soc-dt-generator",
    "gtaylormb/fpga_mpu401",
    "myriadrf/LimeSDR-QPCIe_GW",
    "xilinx/embeddedsw",
]


def search_vhdl_repos_by_stars(max_repos=500):
    """Use gh CLI to find top VHDL repositories sorted by stars."""
    repos = []
    per_page = 100
    pages_needed = (max_repos + per_page - 1) // per_page

    for page in range(1, pages_needed + 1):
        remaining = max_repos - len(repos)
        limit = min(per_page, remaining)
        if limit <= 0:
            break
        cmd = [
            "gh", "search", "repos",
            "--language=VHDL",
            "--sort=stars",
            "--order=desc",
            f"--limit={limit}",
            "--json=fullName,stargazersCount",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            batch = json.loads(result.stdout)
            repos.extend(batch)
            print(f"  [stars] Found {len(batch)} repos (page {page}, total {len(repos)})")
            if len(batch) < limit:
                break
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"  Warning: gh search failed on page {page}: {e}")
            break

    return repos


def search_vhdl_repos_by_activity(max_repos=500):
    """Search for recently updated VHDL repos."""
    repos = []
    per_page = 100
    pages_needed = (max_repos + per_page - 1) // per_page

    for page in range(1, pages_needed + 1):
        remaining = max_repos - len(repos)
        limit = min(per_page, remaining)
        if limit <= 0:
            break
        cmd = [
            "gh", "search", "repos",
            "--language=VHDL",
            "--sort=updated",
            "--order=desc",
            f"--limit={limit}",
            "--json=fullName,stargazersCount",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            batch = json.loads(result.stdout)
            repos.extend(batch)
            print(f"  [activity] Found {len(batch)} repos (page {page}, total {len(repos)})")
            if len(batch) < limit:
                break
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"  Warning: gh search (activity) failed on page {page}: {e}")
            break

    return repos


def search_vhdl_keyword_repos(max_repos=200):
    """Search for repos with VHDL-related keywords that might not be tagged as VHDL language."""
    keywords = ["vhdl fpga", "vhdl entity architecture", "vhdl testbench", "opencores vhdl"]
    repos = []
    per_keyword = max_repos // len(keywords)

    for keyword in keywords:
        cmd = [
            "gh", "search", "repos",
            keyword,
            "--sort=stars",
            "--order=desc",
            f"--limit={per_keyword}",
            "--json=fullName,stargazersCount",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            batch = json.loads(result.stdout)
            repos.extend(batch)
            print(f"  [keyword: {keyword}] Found {len(batch)} repos")
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"  Warning: keyword search '{keyword}' failed: {e}")

    return repos


def gather_all_repos(max_repos=1000):
    """Combine all repo sources, deduplicate, return unique list."""
    all_repos = []

    print("Searching by stars...")
    all_repos.extend(search_vhdl_repos_by_stars(max_repos // 2))

    print("Searching by recent activity...")
    all_repos.extend(search_vhdl_repos_by_activity(max_repos // 3))

    print("Searching by keywords...")
    all_repos.extend(search_vhdl_keyword_repos(max_repos // 5))

    # Add known repos
    print(f"Adding {len(KNOWN_VHDL_REPOS)} known VHDL repos...")
    for name in KNOWN_VHDL_REPOS:
        all_repos.append({"fullName": name, "stargazersCount": 0})

    # Deduplicate by fullName
    seen = set()
    unique = []
    for r in all_repos:
        name = r["fullName"]
        if name not in seen:
            seen.add(name)
            unique.append(r)

    print(f"Total unique repos: {len(unique)}")
    return unique


def clone_repo(full_name, clone_dir):
    """Shallow clone a repo. Returns clone path or None on failure."""
    repo_dir = os.path.join(clone_dir, full_name.replace("/", "__"))
    if os.path.exists(repo_dir):
        return repo_dir
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", f"https://github.com/{full_name}.git", repo_dir],
            capture_output=True, text=True, check=True, timeout=120,
        )
        return repo_dir
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"  Failed to clone {full_name}: {e}")
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir, ignore_errors=True)
        return None


def extract_vhdl_files(clone_dir):
    """Walk clone directory and extract valid VHDL files. Returns list of texts."""
    documents = []
    seen_hashes = set()

    for root, _dirs, files in os.walk(clone_dir):
        for fname in files:
            if not fname.lower().endswith(VHDL_EXTENSIONS):
                continue
            filepath = os.path.join(root, fname)

            # Skip very large files (likely generated/synthesized output)
            try:
                fsize = os.path.getsize(filepath)
                if fsize > MAX_FILE_SIZE:
                    continue
            except OSError:
                continue

            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except (IOError, OSError):
                continue

            # Filter: must be valid UTF-8, at least MIN_LINES lines
            lines = text.splitlines()
            if len(lines) < MIN_LINES:
                continue

            # Basic quality filter: should contain at least one VHDL keyword
            text_lower = text.lower()
            has_vhdl_keyword = any(kw in text_lower for kw in
                ["entity", "architecture", "process", "signal", "library", "port"])
            if not has_vhdl_keyword:
                continue

            # Deduplicate by content hash
            h = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            documents.append(text)

    return documents


def write_parquet_shards(documents, data_dir, shard_size=SHARD_SIZE):
    """Write documents to numbered parquet shards. Shuffles before sharding."""
    os.makedirs(data_dir, exist_ok=True)

    # Clear existing shards
    for f in os.listdir(data_dir):
        if f.startswith("shard_") and f.endswith(".parquet"):
            os.remove(os.path.join(data_dir, f))

    # Shuffle for better train/val split
    random.seed(42)
    random.shuffle(documents)

    num_shards = (len(documents) + shard_size - 1) // shard_size
    for i in range(num_shards):
        batch = documents[i * shard_size : (i + 1) * shard_size]
        table = pa.table({"text": batch})
        path = os.path.join(data_dir, f"shard_{i:05d}.parquet")
        pq.write_table(table, path)

    print(f"Wrote {num_shards} shards ({len(documents)} documents) to {data_dir}")
    return num_shards


def main():
    parser = argparse.ArgumentParser(description="Collect VHDL source code from GitHub")
    parser.add_argument("--max-repos", type=int, default=1000, help="Max repos to search (default: 1000)")
    parser.add_argument("--skip-clone", action="store_true", help="Only process already-cloned repos")
    args = parser.parse_args()

    os.makedirs(CLONE_DIR, exist_ok=True)

    if not args.skip_clone:
        print("Gathering VHDL repositories from multiple sources...")
        repos = gather_all_repos(args.max_repos)

        print(f"\nCloning {len(repos)} repos...")
        cloned = 0
        for i, repo in enumerate(repos):
            name = repo["fullName"]
            result = clone_repo(name, CLONE_DIR)
            if result:
                cloned += 1
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{len(repos)} ({cloned} cloned)")
        print(f"Cloned {cloned}/{len(repos)} repos")

    print("\nExtracting VHDL files...")
    documents = extract_vhdl_files(CLONE_DIR)
    print(f"Extracted {len(documents)} unique VHDL files")

    total_bytes = sum(len(d.encode("utf-8")) for d in documents)
    print(f"Total corpus size: {total_bytes / 1024 / 1024:.1f} MB")

    if not documents:
        print("No VHDL files found! Check your gh CLI authentication.")
        return

    print(f"\nWriting parquet shards...")
    num_shards = write_parquet_shards(documents, DATA_DIR)

    print(f"\nDone! {num_shards} shards in {DATA_DIR}")
    print(f"Total: {len(documents)} files, {total_bytes / 1024 / 1024:.1f} MB")
    print(f"Next: python prepare.py --retrain")


if __name__ == "__main__":
    main()
