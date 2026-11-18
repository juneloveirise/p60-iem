"""Check exported FIRs against physical channel routing and the peak bound."""
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve

import iem
from iem import (
    _blend_filters_with_bound,
    _matrix_l1_bound,
    _matrix_l1_rows,
    bundle,
    design,
    read_curve,
    read_inputs,
    read_measurement,
    read_phase,
)

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


def exported_matrix_bounds(result):
    with zipfile.ZipFile(io.BytesIO(bundle(result))) as archive:
        filters = {"text": {}, "wav": {}}
        for prefix in ("", "base_", "wide_"):
            for name in ("direct", "cross"):
                key = prefix + name
                filters["text"][key] = np.loadtxt(
                    io.BytesIO(archive.read(key + ".txt"))
                )
                wav_rate, values = wavfile.read(io.BytesIO(archive.read(key + ".wav")))
                assert wav_rate == result["rate"]
                filters["wav"][key] = values.astype(np.float64)
        return {
            f"{precision}:{prefix or 'static'}": sum(
                np.abs(filters[precision][prefix + name]).sum(dtype=np.float64)
                for name in ("direct", "cross")
            )
            for precision in ("text", "wav")
            for prefix in ("", "base_", "wide_")
        }


def baseline_design(measurement, phase, rate):
    blend = iem._blend_filters_with_bound
    try:
        iem._blend_filters_with_bound = lambda baseline, candidate, weight: (
            baseline, False, False
        )
        return design(measurement, phase, rate, demo=True)
    finally:
        iem._blend_filters_with_bound = blend


# Coverage interpolation operates on the already-gained FIRs and rejects a
# mixed result if any static, base, or wide TEXT/float32 row exceeds baseline.
guard_baseline = {
    "eq_only": np.zeros(17),
    **{prefix + name: np.zeros(17)
       for prefix in ("", "base_", "wide_")
       for name in ("direct", "cross")},
}
guard_baseline["direct"][8] = 0.2
guard_baseline["cross"][8] = 0.1
guard_baseline["base_direct"][8] = 0.125
guard_baseline["base_cross"][8] = 0.125
guard_baseline["wide_direct"][8] = 0.25
guard_baseline["wide_cross"][8] = 0.15
guard_candidate = {name: values * 0.5 for name, values in guard_baseline.items()}
guard_mix, guard_applied, guard_fallback = _blend_filters_with_bound(
    guard_baseline, guard_candidate, 0.75
)
assert guard_applied and not guard_fallback
np.testing.assert_allclose(guard_mix["direct"], guard_baseline["direct"] * 0.625)
assert _matrix_l1_bound(guard_mix) <= _matrix_l1_bound(guard_baseline)
unsafe_candidate = {name: values.copy() for name, values in guard_baseline.items()}
unsafe_candidate["base_cross"][0] = 0.1
unsafe_mixed = {
    name: values + 0.75 * (unsafe_candidate[name] - values)
    for name, values in guard_baseline.items()
}
assert _matrix_l1_bound(unsafe_mixed) <= _matrix_l1_bound(guard_baseline)
assert _matrix_l1_rows(unsafe_mixed)["float64:base_"] > _matrix_l1_rows(
    guard_baseline
)["float64:base_"]
guard_mix, guard_applied, guard_fallback = _blend_filters_with_bound(
    guard_baseline, unsafe_candidate, 0.75
)
assert not guard_applied and guard_fallback and guard_mix is guard_baseline

# The pre-change FIRs for this fixed synthetic low-bass excess set the reference
# gain and complete-row bounds for each output rate.
baseline_preamp = {44100: -8.2, 48000: -8.2, 96000: -7.5}
baseline_bound = {
    44100: 0.8887239183519393,
    48000: 0.8811276642642343,
    96000: 0.8826552972498707,
}
baseline_margin = {
    44100: 1.0246626289181495,
    48000: 1.0992232649318001,
    96000: 1.0841773606135159,
}
baseline_preview_error = {
    44100: 5.06460652643852,
    48000: 4.982158651353127,
    96000: 4.367186587230363,
}
low_bass_hz = np.geomspace(20, 20000, 240)
low_bass_db = np.where(low_bass_hz < 120, 6.0, 0.0)
low_bass_measurement = (low_bass_hz, low_bass_db)

for rate in (44100, 48000, 96000):
    corrected = design(low_bass_measurement, rate=rate, demo=True)
    assert corrected["bass_cut_extension_coverage_weight"] == 1
    assert corrected["preamp_db"] == baseline_preamp[rate]
    baseline_result = baseline_design(low_bass_measurement, None, rate)
    baseline_rows = exported_matrix_bounds(baseline_result)
    final_rows = exported_matrix_bounds(corrected)
    assert set(final_rows) == set(baseline_rows)
    assert max(baseline_rows.values()) == baseline_bound[rate]
    assert all(final_rows[key] <= baseline_rows[key] + 1e-12 for key in baseline_rows)
    assert corrected["sample_peak_margin_db"] >= baseline_margin[rate] - 1e-12
    assert corrected["bass_cut_extension_applied"] is (rate == 96000)
    assert corrected["bass_cut_extension_headroom_fallback"] is (rate != 96000)
    np.testing.assert_allclose(corrected["latency_ms"], 1000 * 4096 / rate)

    preview_error = corrected["preview"][0, 3] - corrected["preview"][0, 2]
    if rate != 96000:
        # Truncation/windowing leaves a residual; compare to the old FIR value.
        np.testing.assert_allclose(
            preview_error,
            baseline_preview_error[rate],
            atol=1e-9,
        )
    else:
        assert abs(preview_error) < abs(baseline_preview_error[rate])

    with zipfile.ZipFile(io.BytesIO(bundle(corrected))) as archive:
        report = json.loads(archive.read("report.json"))
        assert report["bass_cut_extension_applied"] == corrected["bass_cut_extension_applied"]
        assert report["bass_cut_extension_coverage_weight"] == 1
        assert report["bass_cut_extension_headroom_fallback"] == (
            rate != 96000
        )
        assert report["fir_center_samples"] == 4096
        if rate == 96000:
            np.testing.assert_allclose(
                report["adaptive_reference_delay_ms"], 1000 * 4768 / rate
            )

    with_phase = design(
        low_bass_measurement,
        (phase_hz, shape_deg),
        rate,
        demo=True,
    )
    time_shifted = design(
        low_bass_measurement,
        (phase_hz, wrapped_deg),
        rate,
        demo=True,
    )
    assert with_phase["bass_cut_extension_applied"] is time_shifted[
        "bass_cut_extension_applied"
    ]
    assert with_phase["bass_cut_extension_headroom_fallback"] is time_shifted[
        "bass_cut_extension_headroom_fallback"
    ]
    for name, values in with_phase["filters"].items():
        np.testing.assert_allclose(time_shifted["filters"][name], values, atol=2e-12, rtol=0)

    baseline_phase = baseline_design(
        low_bass_measurement, (phase_hz, shape_deg), rate
    )
    sample_hz = np.array([30, 60, 120, 240, 500, 1000, 2000, 4000])
    offsets = np.arange(8193) - 4096

    def centered_response(values, hz):
        return np.array([
            np.sum(values * np.exp(-2j * np.pi * frequency * offsets / rate))
            for frequency in hz
        ])

    before = centered_response(baseline_phase["filters"]["eq_only"], sample_hz)
    after = centered_response(with_phase["filters"]["eq_only"], sample_hz)
    phase_delta = np.angle(after / before, deg=True)
    assert np.max(np.abs(phase_delta)) < 0.001

# A shorter covered band fades the candidate, while no lower-end data or a
# low-frequency boost leaves the current correction selected.
partial_hz = np.geomspace(22, 20000, 240)
partial = design(
    (partial_hz, np.where(partial_hz < 120, 6.0, 0.0)), rate=96000, demo=True
)
expected_weight = 0.5 * (
    1 + np.cos(np.pi * np.clip(np.log2(22 / 20) / 0.5, 0, 1))
)
np.testing.assert_allclose(partial["bass_cut_extension_coverage_weight"], expected_weight)
assert partial["bass_cut_extension_applied"]

threshold_hz = np.geomspace(20 * np.sqrt(2), 20000, 240)
threshold = design(
    (threshold_hz, np.where(threshold_hz < 120, 6.0, 0.0)), rate=96000, demo=True
)
assert threshold["bass_cut_extension_coverage_weight"] == 0
assert not threshold["bass_cut_extension_applied"]
uncovered_hz = np.geomspace(55, 20000, 240)
uncovered = design(
    (uncovered_hz, np.where(uncovered_hz < 120, 6.0, 0.0)), rate=96000, demo=True
)
assert uncovered["bass_cut_extension_coverage_weight"] == 0
assert not uncovered["bass_cut_extension_applied"]

boosted = design(
    (low_bass_hz, np.where(low_bass_hz < 120, -6.0, 0.0)),
    rate=96000,
    demo=True,
)
assert boosted["bass_cut_extension_coverage_weight"] == 1
assert not boosted["bass_cut_extension_applied"]
assert not boosted["bass_cut_extension_headroom_fallback"]

print("Negative bass extension passes coverage, fixed-gain, TEXT/WAV headroom, and phase-delay checks")
