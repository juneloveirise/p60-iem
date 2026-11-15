"""Run the P60 real-time profile exported beside this script."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", nargs="?", default="camilladsp")
    args = parser.parse_args()
    resolved = shutil.which(str(Path(args.engine).expanduser()))
    if resolved is None:
        parser.error("Provide the path to CamillaDSP with the WideAdaptive patch.")
    engine = str(Path(resolved).resolve())
    folder = Path(__file__).resolve().parent
    config = folder/"camilladsp-realtime.yml"
    try:
        devices = json.loads(config.read_text(encoding="utf-8"))["devices"]
    except (OSError, ValueError, KeyError) as exc:
        parser.error(f"Missing or invalid real-time export: {exc}")
    expected = {"Darwin": "CoreAudio","Windows": "Wasapi"}.get(platform.system())
    if expected is None or any(devices[k]["type"] != expected for k in ("capture","playback")):
        parser.error("Use the real-time export for this operating system.")
    subprocess.run([engine,"--check",str(config)],cwd=folder,check=True,timeout=20)
    print("P60 running in foreground. Press Ctrl+C to stop.",flush=True)
    try:
        return subprocess.run([engine,str(config)],cwd=folder).returncode
    except KeyboardInterrupt:
        return 130

if __name__ == "__main__":
    raise SystemExit(main())
