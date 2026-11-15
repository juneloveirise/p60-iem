"""Check exported FIRs against physical channel routing and the peak bound."""
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve

from iem import bundle, design, read_curve

fixtures = Path(__file__).parent / "examples"
measurement, phase = [
    read_curve((fixtures / f"{name}.csv").read_text())
    for name in ("measurement", "phase")
]
rng = np.random.default_rng(4138)
signal = rng.uniform(-1, 1, (2, 12000))
for rate in (44100, 48000, 96000):
    for phase_input in (None, phase):
        result = design(measurement, phase_input, rate, demo=True)
        with zipfile.ZipFile(io.BytesIO(bundle(result))) as archive:
            filters = {}
            for name in ("eq_only", "direct", "cross"):
                audio_rate, impulse = wavfile.read(io.BytesIO(archive.read(name+".wav")))
                assert audio_rate == rate and np.isfinite(impulse).all()
                filters[name] = impulse.astype(float)
            eq, direct, cross = (filters[key] for key in ("eq_only", "direct", "cross"))
            np.testing.assert_allclose(direct+cross, eq, atol=5e-8)
            assert np.max(abs(cross)) > 1e-5
            left = fftconvolve(signal[0], direct)+fftconvolve(signal[1], cross)
            right = fftconvolve(signal[1], direct)+fftconvolve(signal[0], cross)
            bound = np.abs(direct).sum()+np.abs(cross).sum()
            assert max(np.max(abs(left)), np.max(abs(right))) <= bound < 10**(-.99/20)
            assert json.loads(archive.read("camilladsp-offline.yml"))["devices"]["samplerate"] == rate
            measured_error = result["preview"][:,1]-result["preview"][:,2]
            corrected_error = result["preview"][:,3]-result["preview"][:,2]
            assert np.linalg.norm(corrected_error) < .5*np.linalg.norm(measured_error)
print("Six combinations passed: three rates and two phase modes")
