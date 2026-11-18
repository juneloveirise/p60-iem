"""Exercise the upload form with synthetic data using Streamlit's AppTest (1.64+)."""
import csv
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from streamlit.testing.v1 import AppTest


def check_export(app):
    assert not app.exception and not app.error, (app.exception, app.error)
    assert len(app.download_button) == 1


def verify():
    source = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        for name in ("app.py", "iem.py", "play.py", "prepare_reference.py", "p60_dsp.csv"):
            shutil.copyfile(source / name, root / name)
        examples = root / "examples"
        shutil.copytree(source / "examples", examples)
        measurement = examples / "measurement.csv"
        phase = examples / "phase.csv"
        # The temporary target exercises the real preparation path with public synthetic data.
        subprocess.run([sys.executable, "prepare_reference.py", str(measurement)], cwd=root, check=True)
        with measurement.open(encoding="utf-8") as magnitudes, phase.open(encoding="utf-8") as phases:
            levels, angles = csv.reader(magnitudes), csv.reader(phases)
            next(levels)
            next(angles)
            rows = ["Hz,dB,Phase (deg)"]
            for (hz, db), (phase_hz, deg) in zip(levels, angles, strict=True):
                assert float(hz) == float(phase_hz)
                rows.append(f"{hz},{db},{deg}")
        combined = "\n".join(rows).encode("utf-8")

        app = AppTest.from_file(str(root / "app.py"), default_timeout=45).run()
        check_export(app)
        app.checkbox[0].uncheck().run()
        assert not app.exception and len(app.file_uploader) == 2 and not app.download_button
        app.file_uploader(key="measurement").upload("combined.csv", combined, "text/csv").run()
        check_export(app)
        assert any("同一文件" in caption.value for caption in app.caption)
        app.file_uploader(key="phase").upload("override.csv", phase.read_bytes(), "text/csv").run()
        check_export(app)
        assert any("单独导入" in caption.value for caption in app.caption)

        backend = next(widget for widget in app.selectbox if widget.label == "音频系统")
        app.text_input(key=f"playback-{backend.value}").set_value("Offline test DAC").run()
        check_export(app)
        app.file_uploader(key="phase").clear().run()
        unsupported = combined.replace(b"Phase (deg)", b"Phase (rad)", 1)
        app.file_uploader(key="measurement").upload("radians.csv", unsupported, "text/csv").run()
        assert not app.exception and len(app.error) == 1 and not app.download_button


if __name__ == "__main__":
    verify()
    print("GUI demo, uploads, override, device configuration form and invalid phase input passed.")
