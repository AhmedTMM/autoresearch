"""
Collect VHDL textbook data for instruction tuning.

Sources:
    - Free Range VHDL (open-source textbook, LaTeX on GitHub)
    - University VHDL example repos (MIT/open-licensed)

Extracts (English explanation, VHDL code) pairs and saves them as:
    1. Raw VHDL parquet shards (for pretraining, goes to ~/.cache/autoresearch/data/)
    2. Instruction pairs JSON (for instruction tuning, goes to ~/.cache/autoresearch/instruction_pairs.json)

Usage:
    uv run collect_textbooks.py
"""

import json
import os
import re
import shutil
import subprocess
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
CLONE_DIR = os.path.join(CACHE_DIR, "textbook_clones")
PAIRS_PATH = os.path.join(CACHE_DIR, "instruction_pairs.json")


def clone_repo(url, name):
    """Clone a git repo into CLONE_DIR, return path."""
    dest = os.path.join(CLONE_DIR, name)
    if os.path.exists(dest):
        print(f"  Already cloned: {name}")
        return dest
    os.makedirs(CLONE_DIR, exist_ok=True)
    print(f"  Cloning {url}...")
    subprocess.run(
        ["git", "clone", "--depth=1", url, dest],
        capture_output=True, timeout=120,
    )
    return dest


def extract_free_range_vhdl(repo_path):
    """Extract (explanation, code) pairs from Free Range VHDL LaTeX source."""
    pairs = []
    raw_vhdl = []

    tex_files = sorted(f for f in os.listdir(repo_path) if f.endswith(".tex"))
    for tex_file in tex_files:
        path = os.path.join(repo_path, tex_file)
        with open(path, "r", errors="replace") as f:
            content = f.read()

        # Find VHDL code blocks in LaTeX
        # Common patterns: \begin{lstlisting} ... \end{lstlisting}
        # or \begin{verbatim} ... \end{verbatim}
        code_blocks = re.findall(
            r'\\begin\{lstlisting\}(.*?)\\end\{lstlisting\}',
            content, re.DOTALL
        )
        code_blocks += re.findall(
            r'\\begin\{verbatim\}(.*?)\\end\{verbatim\}',
            content, re.DOTALL
        )

        for code in code_blocks:
            code = code.strip()
            # Only keep blocks that look like VHDL
            code_lower = code.lower()
            if not any(kw in code_lower for kw in ["library", "entity", "architecture", "process", "signal", "port"]):
                continue
            if len(code) < 30:
                continue

            raw_vhdl.append(code)

            # Try to find preceding explanation
            # Look for text before this code block
            idx = content.find(code)
            if idx > 0:
                # Get ~500 chars before the code block
                pre_text = content[max(0, idx - 500):idx]
                # Strip LaTeX commands, keep plain text
                explanation = clean_latex(pre_text)
                explanation = explanation.strip()
                # Get last 1-2 sentences
                sentences = re.split(r'[.!?]\s+', explanation)
                sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
                if sentences:
                    explanation = ". ".join(sentences[-2:]) + "."
                    if len(explanation) > 20:
                        pairs.append((explanation, code))

    print(f"  Free Range VHDL: {len(raw_vhdl)} code blocks, {len(pairs)} instruction pairs")
    return pairs, raw_vhdl


def clean_latex(text):
    """Strip LaTeX commands, keep plain text."""
    # Remove comments
    text = re.sub(r'%.*?\n', ' ', text)
    # Remove common LaTeX commands
    text = re.sub(r'\\(textbf|textit|emph|underline|texttt)\{([^}]*)\}', r'\2', text)
    text = re.sub(r'\\(section|subsection|subsubsection|chapter|paragraph)\*?\{([^}]*)\}', r'\2', text)
    text = re.sub(r'\\(label|ref|cite|index)\{[^}]*\}', '', text)
    text = re.sub(r'\\(begin|end)\{[^}]*\}', '', text)
    text = re.sub(r'\\item\s*', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    # Remove braces
    text = re.sub(r'[{}]', '', text)
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_vhdl_from_repo(repo_path, repo_name):
    """Extract VHDL files from a repo, create instruction pairs from entity declarations."""
    pairs = []
    raw_vhdl = []

    for root, dirs, files in os.walk(repo_path):
        # Skip hidden dirs and test dirs
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in files:
            if not fname.lower().endswith((".vhd", ".vhdl")):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", errors="replace") as f:
                    code = f.read()
            except Exception:
                continue

            if len(code) < 50 or len(code) > 20000:
                continue
            if "library" not in code.lower():
                continue

            raw_vhdl.append(code)

            # Try to extract entity name and create a description
            entity_match = re.search(
                r'entity\s+(\w+)\s+is\s*\n?\s*(port\s*\(.*?\);)?',
                code, re.DOTALL | re.IGNORECASE
            )
            if entity_match:
                entity_name = entity_match.group(1)
                # Create a natural description from the entity name
                readable = entity_name.replace("_", " ")
                pairs.append((f"Design {readable} in VHDL", code))

    print(f"  {repo_name}: {len(raw_vhdl)} VHDL files, {len(pairs)} instruction pairs")
    return pairs, raw_vhdl


# Repos to collect from (all open-source / permissively licensed)
REPOS = [
    ("https://github.com/fabriziotappero/Free-Range-VHDL-book.git", "free-range-vhdl", "textbook"),
    ("https://github.com/fcayci/vhdl-digital-design.git", "vhdl-digital-design", "examples"),
    ("https://github.com/khaledhassan/vhdl-examples.git", "vhdl-examples", "examples"),
    ("https://github.com/tomas-fryza/vhdl-labs.git", "vhdl-labs", "examples"),
]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    all_pairs = []
    all_raw_vhdl = []

    for url, name, repo_type in REPOS:
        try:
            repo_path = clone_repo(url, name)
        except Exception as e:
            print(f"  Failed to clone {name}: {e}")
            continue

        if repo_type == "textbook":
            pairs, raw = extract_free_range_vhdl(repo_path)
        else:
            pairs, raw = extract_vhdl_from_repo(repo_path, name)

        all_pairs.extend(pairs)
        all_raw_vhdl.extend(raw)

    # Also look for README-based examples in the example repos
    for url, name, repo_type in REPOS:
        if repo_type != "examples":
            continue
        repo_path = os.path.join(CLONE_DIR, name)
        if not os.path.exists(repo_path):
            continue
        # Check for README files with VHDL code blocks
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fname in files:
                if fname.lower() in ("readme.md", "readme.rst", "readme.txt"):
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, "r", errors="replace") as f:
                            content = f.read()
                    except Exception:
                        continue
                    # Extract markdown code blocks with VHDL
                    blocks = re.findall(r'```(?:vhdl)?\n(.*?)```', content, re.DOTALL | re.IGNORECASE)
                    for block in blocks:
                        if any(kw in block.lower() for kw in ["library", "entity", "architecture"]):
                            all_raw_vhdl.append(block)

    print(f"\nTotal: {len(all_raw_vhdl)} raw VHDL blocks, {len(all_pairs)} instruction pairs")

    # Save instruction pairs as JSON
    print(f"Saving instruction pairs to {PAIRS_PATH}")
    with open(PAIRS_PATH, "w") as f:
        json.dump(all_pairs, f, indent=2)

    # Save raw VHDL as a parquet shard (for pretraining)
    if all_raw_vhdl:
        shard_path = os.path.join(DATA_DIR, "shard_textbooks.parquet")
        table = pa.table({"text": all_raw_vhdl})
        pq.write_table(table, shard_path)
        print(f"Saved {len(all_raw_vhdl)} VHDL blocks to {shard_path}")

    print("Done!")


if __name__ == "__main__":
    main()
