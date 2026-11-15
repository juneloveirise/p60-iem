"""Reconstruct the fixed P60 target from an EPZ P50 measurement."""
import argparse
import json
from pathlib import Path
import numpy as np
from iem import log_interpolate, read_curve

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurement",type=Path,help="Hui EPZ P50 RAW_FR JSON or Hz/dB text")
    args = parser.parse_args()
    text = args.measurement.read_text(encoding="utf-8-sig")
    if text.lstrip().startswith("{"):
        record = json.loads(text)["data"]
        if record.get("uuid") != "185a3d4a-fc5b-458b-b0a5-db4bb716d7cf":
            parser.error("Expected Hui EPZ P50 measurement")
        raw, = [entry for entry in record["datas"] if entry["project"]["code"]=="RAW_FR"]
        text = "\n".join(",".join(map(str,row)) for row in raw["children"][0]["dataSet"] if len(row)==2)
    measured = read_curve(text)
    if measured[0,0] > 20.01 or measured[0,-1] < 19800:
        parser.error("The P50 measurement must cover approximately 20 Hz to 20 kHz")
    dsp = np.loadtxt(ROOT/"p60_dsp.csv",delimiter=",",skiprows=1)
    level = log_interpolate(measured,dsp[:,0])
    dsp[:,1] += level - np.interp(500,dsp[:,0],level)
    np.savetxt(ROOT/"p60_reference.csv",dsp,delimiter=",",comments="",fmt="%.10g",
               header="hz,target_db,base_real,base_imag,wide_real,wide_imag")
    print("Fixed P60 reference prepared locally")

if __name__ == "__main__":
    main()
