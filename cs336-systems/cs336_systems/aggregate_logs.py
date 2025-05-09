# aggregator.py
import glob
import re
import csv

# Patterns to extract metadata from filename and results from log content
LOG_GLOB = "logs/bench_*.log"
HEADER_RE = re.compile(r"=== Model=(\w+) Mode=(\w+) Precision=(\w+) Warmup=(\d+) ===")
RESULT_RE = re.compile(r"\|\s*mean\s*([\d\.]+)s\s*±\s*([\d\.]+)s")

def parse_log(path):
    model = mode = precision = warmup = None
    mean = std = None
    with open(path, 'r') as f:
        for line in f:
            if model is None:
                m = HEADER_RE.search(line)
                if m:
                    model, mode, precision, warmup = m.groups()
                    continue
            if mean is None:
                r = RESULT_RE.search(line)
                if r:
                    mean, std = r.groups()
                    break
    if None in (model, mode, precision, warmup, mean, std):
        raise ValueError(f"Failed to parse {path}")
    return {
        "model": model,
        "mode": mode,
        "precision": precision,
        "warmup": int(warmup),
        "mean_s": float(mean),
        "std_s": float(std),
    }

def main():
    rows = []
    for path in glob.glob(LOG_GLOB):
        try:
            data = parse_log(path)
            rows.append(data)
        except ValueError as e:
            print(e)
    # Write CSV
    with open("benchmark_results.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "model", "mode", "precision", "warmup", "mean_s", "std_s"
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("Results aggregated into benchmark_results.csv")

if __name__ == "__main__":
    main()
