# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 19:44:21 2025

@author: dac00
"""
import os
import glob
import csv
import re

def parse_file(filepath):
    """Parse one file and return a dict of metric -> value."""
    data = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()

            # remove brackets if present
            val = val.replace("[", "").replace("]", "")

            # convert to float if possible
            try:
                val = float(val)
            except ValueError:
                continue
            data[key] = val
    return data

def parse_filename(fname):
    """
    Extract parameters from filename like:
    MSE_0.2_fieldMeas_9_T7_0.1.txt
    Returns dict with MSE_val, fieldMeas_val, T, extra.
    """
    base = os.path.basename(fname)
    m = re.match(r"MSE_([0-9.]+)_fieldMeas_([0-9]+)_T([0-9]+)_([0-9.]+)\.txt", base)
    if not m:
        return {}
    mse_val = float(m.group(1))
    fieldmeas_val = int(m.group(2))
    t_val = int(m.group(3))
    extra_val = float(m.group(4))
    return {
        "fieldNum": fieldmeas_val,
        "T": t_val,
        "velVariance": extra_val
    }

def main(input_pattern="MSE_*_fieldMeas_*_T*_*.txt", output_csv="results.csv"):
    files = sorted(glob.glob(input_pattern))
    if not files:
        print("No matching files found.")
        return

    rows = []
    for filepath in files:
        row = {"filename": os.path.basename(filepath)}
        # add metrics from content
        row.update(parse_file(filepath))
        # add parameters from filename
        row.update(parse_filename(filepath))
        rows.append(row)

    # Collect all keys for CSV header
    headers = set()
    for r in rows:
        headers.update(r.keys())
    headers = ["filename"] + sorted([h for h in headers if h != "filename"])

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")

if __name__ == "__main__":
    main()
