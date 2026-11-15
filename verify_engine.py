"""Exercise the spawned adaptive engine without accessing audio devices."""
import json
import platform
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from scipy.signal import fftconvolve

from iem import adaptive_config, design, read_curve, realtime_config

engine = Path(sys.argv[1]).resolve()
fixtures = Path(__file__).parent/"examples"
measurement = read_curve((fixtures/"measurement.csv").read_text())
filters = design(measurement, demo=True)["filters"]
rate = 96000
delay = 672
length = 32768
bins = np.fft.rfftfreq(length,1/rate)
band = (bins>=2500)&(bins<=5000)
direct_base, cross_base, direct_wide, cross_wide = [
    np.fft.rfft(filters[key],length)[band]
    for key in ("base_direct","base_cross","wide_direct","wide_cross")
]
# Choose a coherent stereo tone with a material wide-endpoint loss.
# A well-conditioned nonzero baseline avoids deriving the test from a null.
candidates = .8*np.exp(1j*np.linspace(-np.pi,np.pi,72,endpoint=False))[:,None]
baseline = abs(direct_base+candidates*cross_base)
endpoint = abs(direct_wide+candidates*cross_wide)
ratio = np.where(baseline>.01,endpoint/np.maximum(baseline,1e-20),np.inf)
angle_index,frequency_index = np.unravel_index(np.argmin(ratio),ratio.shape)
frequency = bins[band][frequency_index]
wave = .65*np.exp(2j*np.pi*frequency*np.arange(length)/rate)
signal = np.column_stack((wave.real,(wave*candidates[angle_index,0]).real)).astype("<f4")
signal[:4096,1] = signal[:4096,0]
signal[-4096:,1] = signal[-4096:,0]
with tempfile.TemporaryDirectory() as folder:
    work = Path(folder)
    for name, h in filters.items():
        np.savetxt(work/(name+".txt"),h,fmt="%.17g")
    backend, capture = {
        "Darwin": ("CoreAudio","BlackHole 2ch"),
        "Windows": ("Wasapi","CABLE Output (VB-Audio Virtual Cable)"),
    }[platform.system()]
    live = realtime_config(backend,capture,"Test headphones")
    config_path = work/"realtime.yml"
    config_path.write_text(json.dumps(live))
    subprocess.run([str(engine),"--check",str(config_path)],cwd=work,check=True,
                   timeout=20,capture_output=True)
    print(f"{backend} real-time configuration passed the engine schema check")
    signal.tofile(work/"input.raw")
    (work/"config.yml").write_text(json.dumps(adaptive_config(rate)))
    for args in (["--check","config.yml"],["config.yml"]):
        subprocess.run([str(engine),*args],cwd=work,check=True,timeout=30,
                       capture_output=True)
    rendered = np.fromfile(work/"output.raw",dtype="<f4").reshape(-1,2)
assert len(rendered)>=length and np.isfinite(rendered).all()
def stereo(prefix):
    direct,cross = (filters[prefix+x] for x in ("direct","cross"))
    def conv(h,x):
        return fftconvolve(x,h)[:length]
    aligned = np.column_stack([
        conv(direct,signal[:,0])+conv(cross,signal[:,1]),
        conv(direct,signal[:,1])+conv(cross,signal[:,0]),
    ])
    return np.pad(aligned,((delay,0),(0,0)))[:length]
base,wide = stereo("base_"),stereo("wide_")
output = rendered[:length].astype(float)
np.testing.assert_array_less(np.minimum(base,wide)-2e-7,output)
np.testing.assert_array_less(output,np.maximum(base,wide)+2e-7)
difference = wide-base
weight = np.sum((output-base)*difference,axis=1)/np.maximum(np.sum(difference**2,axis=1),1e-20)
active = np.max(abs(difference),axis=1)>1e-4
assert active.sum()>1000
np.testing.assert_allclose(output[active],(base+weight[:,None]*difference)[active],atol=2e-7)
assert np.ptp(weight[active])>.1
print(f"Adaptive engine passed: shared weight {min(weight[active]):.3f}–{max(weight[active]):.3f}; peak {np.max(abs(output)):.6f}")
