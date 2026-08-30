from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess


def run_model(nusmv: Path, model: Path) -> dict:
    proc = subprocess.run([str(nusmv), str(model)], text=True, capture_output=True, check=False)
    text = proc.stdout + "\n" + proc.stderr
    specs = []
    for line in text.splitlines():
        m = re.search(r"-- specification (.+) is (true|false)", line, re.I)
        if m:
            specs.append({"specification": m.group(1).strip(), "result": m.group(2).lower() == "true"})
    return {"returncode": proc.returncode, "specifications": specs, "raw_output": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nusmv", type=Path, required=True)
    ap.add_argument("--secured", type=Path, default=Path("formal/secured.smv"))
    ap.add_argument("--negative", type=Path, default=Path("formal/negative_control.smv"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/formal_results.json"))
    args = ap.parse_args()
    payload = {
        "secured": run_model(args.nusmv, args.secured),
        "negative_control": run_model(args.nusmv, args.negative),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: {"returncode": v["returncode"], "specifications": v["specifications"]} for k, v in payload.items()}, indent=2))
    if payload["secured"]["returncode"] != 0 or not payload["secured"]["specifications"]:
        raise SystemExit("Secured NuSMV model did not verify successfully")


if __name__ == "__main__":
    main()
