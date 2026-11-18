"""Check exported FIRs against physical channel routing and the peak bound."""
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve

from iem import bundle, design, read_curve, read_inputs, read_measurement, read_phase

fixtures = Path(__file__).parent / "examples"
measurement, phase = [
    read_curve((fixtures / f"{name}.csv").read_text())
    for name in ("measurement", "phase")
]

test_frequencies = np.geomspace(20, 18000, 24)
combined_rows = [
    f"{frequency:.15g},{-2.0:.1f},{index * 7 % 181}"
    for index, frequency in enumerate(test_frequencies)
]
combined = "Frequency (Hz),Level (dB),Phase (degrees)\n" + "\n".join(combined_rows)
combined_measurement, combined_phase = read_measurement(combined)
np.testing.assert_allclose(combined_measurement[0], test_frequencies)
np.testing.assert_allclose(combined_measurement[1], -2)
np.testing.assert_allclose(combined_phase[1], np.arange(len(test_frequencies)) * 7 % 181)

unlabeled_extra = "Hz,dB,Coherence\n" + "\n".join(
    f"{frequency:.9g},-3,0.97" for frequency in test_frequencies
)
extra_measurement, extra_phase = read_measurement(unlabeled_extra)
np.testing.assert_allclose(extra_measurement[1], -3)
assert extra_phase is None

raw_three_column = "\n".join(
    f"{frequency:.9g},-4,123" for frequency in test_frequencies
)
raw_measurement, raw_phase = read_measurement(raw_three_column)
np.testing.assert_allclose(raw_measurement[1], -4)
assert raw_phase is None
np.testing.assert_allclose(read_curve(raw_three_column, column=2)[1], 123)

for label in ("Phase coherence", "Phase quality", "Right channel"):
    labeled_quality = f"Hz,dB,{label}\n" + "\n".join(
        f"{frequency:.9g},-3,0.97" for frequency in test_frequencies
    )
    _, quality_phase = read_measurement(labeled_quality)
    assert quality_phase is None

for label in ("Phase coherence", "Phase quality", "Right channel"):
    whitespace_columns = (
        f"Freq (Hz) SPL (dB) {label}\n"
        + "\n".join(f"{frequency:.15g} -3 0.97" for frequency in test_frequencies)
    )
    _, whitespace_phase = read_measurement(whitespace_columns)
    assert whitespace_phase is None

khz_measurement = "Frequency(kHz),SPL(dB)\n" + "\n".join(
    f"{frequency / 1000:.15g},-3" for frequency in test_frequencies
)
converted_measurement, converted_phase = read_measurement(khz_measurement)
np.testing.assert_allclose(converted_measurement[0], test_frequencies)
assert converted_phase is None
khz_phase = "Frequency(kHz),Phase\n" + "\n".join(
    f"{frequency / 1000:.15g},25" for frequency in test_frequencies
)
converted_phase_curve = read_phase(khz_phase)
np.testing.assert_allclose(converted_phase_curve[0], test_frequencies)
np.testing.assert_allclose(converted_phase_curve[1], 25)

rew_measurement = (
    "* REW export metadata\n"
    "* Freq(Hz) SPL(dB) Phase(degrees)\n"
    + "\n".join(
        f"{frequency:.15g} -3 {index}" for index, frequency in enumerate(test_frequencies)
    )
)
rew_curve, rew_phase = read_measurement(rew_measurement)
np.testing.assert_allclose(rew_curve[0], test_frequencies)
np.testing.assert_allclose(rew_phase[1], np.arange(len(test_frequencies)))

spaced_header = (
    "Freq (Hz) SPL (dB) Phase (deg)\n"
    + "\n".join(
        f"{frequency:.15g} -5 {index - 10}"
        for index, frequency in enumerate(test_frequencies)
    )
)
spaced_curve, spaced_phase = read_measurement(spaced_header)
np.testing.assert_allclose(spaced_curve[1], -5)
np.testing.assert_allclose(spaced_phase[1], np.arange(len(test_frequencies)) - 10)

separate_phase_text = "Hz,Degrees\n" + "\n".join(
    f"{frequency:.9g},45" for frequency in test_frequencies
)
chosen_measurement, chosen_phase, phase_source = read_inputs(combined, separate_phase_text)
assert phase_source == "separate"
np.testing.assert_allclose(chosen_measurement, combined_measurement)
np.testing.assert_allclose(chosen_phase[1], 45)

legacy_phase_text = "Hz,Phase\n" + "\n".join(
    f"{frequency:.15g},45" for frequency in test_frequencies
)
legacy_phase_curve = read_phase(legacy_phase_text)
np.testing.assert_allclose(legacy_phase_curve[1], 45)

phase_only_measurement = "Hz,Phase (deg)\n" + "\n".join(
    f"{frequency:.15g},45" for frequency in test_frequencies
)
try:
    read_inputs(phase_only_measurement, legacy_phase_text)
except ValueError:
    pass
else:
    raise AssertionError("A phase-only table was accepted as a dB measurement")

radian_header = "Hz,dB,Phase (rad)\n" + "\n".join(
    f"{frequency:.9g},-2,0.5" for frequency in test_frequencies
)
try:
    read_measurement(radian_header)
except ValueError as exc:
    assert "弧度" in str(exc)
else:
    raise AssertionError("Radian phase column was accepted as degrees")
_, radian_override, override_source = read_inputs(radian_header, separate_phase_text)
assert override_source == "separate"
np.testing.assert_allclose(radian_override[1], 45)

radian_phase_file = "Frequency (Hz),Phase (rad)\n" + "\n".join(
    f"{frequency:.15g},0.5" for frequency in test_frequencies
)
try:
    read_phase(radian_phase_file)
except ValueError as exc:
    assert "弧度" in str(exc)
else:
    raise AssertionError("Radian phase input was accepted as degrees")

for wrong_phase_header in ("Hz,Radians", "Hz,dB"):
    wrong_phase_file = wrong_phase_header + "\n" + "\n".join(
        f"{frequency:.15g},10" for frequency in test_frequencies
    )
    try:
        read_phase(wrong_phase_file)
    except ValueError:
        pass
    else:
        raise AssertionError(f"{wrong_phase_header} was accepted as degree phase")

invalid_tables = (
    "Hz,Phase\n" + "\n".join(
        f"{frequency:.15g},{index}" for index, frequency in enumerate(test_frequencies)
    ),
    "Hz,Phase (deg)\n" + "\n".join(
        f"{frequency:.15g},{index}" for index, frequency in enumerate(test_frequencies)
    ),
    "Hz,dB\n" + "\n".join(f"{frequency:.9g},nan" for frequency in test_frequencies),
    "Hz,dB\n" + "\n".join(
        f"{frequency:.9g},{index}" for index, frequency in enumerate(test_frequencies)
    ) + f"\n{test_frequencies[0]:.9g},99",
    "Hz,dB\n" + "\n".join(
        f"{frequency:.9g},-3" for frequency in test_frequencies[:7]
    ),
    "Hz,dB,Phase\n" + "\n".join(
        f"{frequency:.9g},-3,10,extra" for frequency in test_frequencies
    ),
)
for invalid_table in invalid_tables:
    try:
        read_measurement(invalid_table)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid measurement input was accepted")

try:
    read_inputs(combined, "")
except ValueError:
    pass
else:
    raise AssertionError("An explicitly supplied empty phase file was ignored")

ambiguous_levels = "Hz,Left dB,Right dB\n" + "\n".join(
    f"{frequency:.9g},-3,-4" for frequency in test_frequencies
)
try:
    read_measurement(ambiguous_levels)
except ValueError as exc:
    assert "多个 dB 列" in str(exc)
else:
    raise AssertionError("Ambiguous two-channel measurement was accepted")

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
            report = json.loads(archive.read("report.json"))
            center = report["fir_center_samples"]
            assert center == 4096
            if rate == 96000:
                assert report["adaptive_lookahead_samples"] == 672
                np.testing.assert_allclose(report["adaptive_reference_delay_ms"], 1000 * 4768 / rate)
            else:
                assert "adaptive_reference_delay_ms" not in report
            response = np.loadtxt(io.BytesIO(archive.read("eq-response.csv")), delimiter=",", skiprows=1)
            text_fir = np.loadtxt(io.BytesIO(archive.read("eq_only.txt")))
            offset = np.arange(len(text_fir)) - center
            for hz, gain_db, phase_deg, delay_ms in response[::32]:
                def centered(at_hz):
                    return np.sum(text_fir * np.exp(-2j * np.pi * at_hz * offset / rate))
                value = centered(hz)
                np.testing.assert_allclose(20 * np.log10(abs(value)), gain_db, atol=1e-8)
                np.testing.assert_allclose(np.exp(1j * np.deg2rad(phase_deg)), value / abs(value), atol=1e-9)
                step_hz = 0.05
                slope = np.angle(centered(hz + step_hz) / centered(hz - step_hz)) / (2 * step_hz)
                np.testing.assert_allclose(-1000 * slope / (2 * np.pi), delay_ms, atol=2e-5)
            measured_error = result["preview"][:,1]-result["preview"][:,2]
            corrected_error = result["preview"][:,3]-result["preview"][:,2]
            assert np.linalg.norm(corrected_error) < .5*np.linalg.norm(measured_error)
print("Six combinations passed: three rates and two phase modes")

# A change to the measurement time origin must leave the exported FIRs intact.
# The irregular spacing distinguishes interpolation in Hz from interpolation in log Hz.
phase_hz = np.unique(np.r_[np.geomspace(20, 20000, 100), 300, 1000])
shape_deg = 15 * np.sin(2 * np.pi * (phase_hz - 800) / 5800)
timed_deg = shape_deg + 40 - 360 * 0.00025 * phase_hz
assert np.max(np.abs(np.diff(timed_deg))) < 180
wrapped_deg = (timed_deg + 180) % 360 - 180
for rate in (44100, 48000, 96000):
    original = design(measurement, (phase_hz, shape_deg), rate, demo=True)
    shifted = design(measurement, (phase_hz, wrapped_deg), rate, demo=True)
    assert original["preamp_db"] == shifted["preamp_db"]
    for name, values in original["filters"].items():
        np.testing.assert_allclose(shifted["filters"][name], values, atol=2e-12, rtol=0)
print("The seven FIRs are invariant to a fixed phase offset and time shift at all three rates")
