"""Offline IEM correction and stereo FIR export."""
import argparse
import csv
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import freqz


def _split_fields(line):
    if "," in line:
        return next(csv.reader([line], skipinitialspace=True))
    if ";" in line:
        return next(csv.reader([line], delimiter=";", skipinitialspace=True))
    if "\t" in line:
        return next(csv.reader([line], delimiter="\t", skipinitialspace=True))
    return re.findall(
        r"(?:phase\s+(?:coherence|quality)|(?:left|right)\s+(?:channel|dB)"
        r"|\S+(?:\s+\([^)]*\))?)",
        line.strip(),
        flags=re.IGNORECASE,
    )


def _header_token(value):
    return re.sub(r"[^a-z0-9]+", "", value.strip().lstrip("#").casefold())


def _looks_like_header(fields):
    tokens = [_header_token(field) for field in fields]
    has_frequency = any("freq" in name or name == "hz" for name in tokens)
    has_value = any(
        "db" in name or "decibel" in name or "spl" in name
        or name in {"deg", "degree", "degrees"}
        or re.fullmatch(r"phase(?:angle)?(?:deg|degree|degrees)", name)
        for name in tokens
    )
    return has_frequency and (has_value or len(fields) == 2)


def _parse_table(text):
    header = None
    rows = []
    width = None
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip().lstrip("\ufeff")
        if not line:
            continue
        if line.startswith(("#", "*")):
            candidate = _split_fields(line[1:].strip())
            if not rows and _looks_like_header(candidate):
                header = candidate
            continue

        fields = _split_fields(line)
        try:
            float(fields[0])
        except ValueError:
            if rows:
                raise ValueError(f"第 {line_number} 行不是有效的测量数据")
            if _looks_like_header(fields):
                header = fields
            continue

        try:
            values = [float(field) for field in fields]
        except ValueError as exc:
            raise ValueError(f"第 {line_number} 行含非数字数据") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"第 {line_number} 行含非有限数值")
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise ValueError(f"第 {line_number} 行的列数与前面数据不一致")
        if header is not None and len(header) != len(values):
            raise ValueError(f"第 {line_number} 行的列数与表头不一致")
        rows.append(values)

    if not rows:
        raise ValueError("文件中没有可用的频率数据")
    return header, rows


def _curve(rows, frequency_column, value_column, frequency_scale=1):
    try:
        data = np.asarray(
            [(row[frequency_column] * frequency_scale, row[value_column]) for row in rows],
            dtype=float,
        )
    except IndexError as exc:
        raise ValueError("每条数据需要 Hz 和指定的数值列") from exc
    data = data[np.argsort(data[:, 0])]
    if (len(data) < 8 or not np.isfinite(data).all()
            or data[0, 0] <= 0 or np.any(np.diff(data[:, 0]) <= 0)):
        raise ValueError("曲线需要至少 8 个唯一且有效的正频率点")
    return data.T


def _frequency_column(tokens, width):
    columns = [i for i, name in enumerate(tokens)
               if name == "hz" or name.startswith("freq")]
    if len(columns) > 1:
        raise ValueError("表头中有多个频率列，请只保留一个")
    column = columns[0] if columns else (0 if width == 2 else None)
    if column is None:
        raise ValueError("多列数据需要用表头标出 Hz 或 Frequency 列")
    name = tokens[column]
    if name == "hz":
        return column, 1
    base = "frequency" if name.startswith("frequency") else "freq"
    unit = name[len(base):]
    if unit not in {"", "hz", "khz", "mhz"}:
        raise ValueError("频率单位请标为 Hz、kHz 或 MHz")
    if unit == "mhz":
        scale = 1_000_000
    elif unit == "khz":
        scale = 1_000
    else:
        scale = 1
    return column, scale


def _phase_columns(tokens, *, require_degree=True, allow_unqualified=False):
    degree_names = {"deg", "degree", "degrees", "angledeg",
                    "angledegree", "angledegrees"}
    columns = []
    for index, name in enumerate(tokens):
        if ("coherence" in name or "quality" in name):
            continue
        if name in degree_names or re.fullmatch(r"phase(?:angle)?(?:deg|degree|degrees)", name):
            columns.append(index)
        elif "phase" in name and ("rad" in name or "db" in name):
            if require_degree:
                if "rad" in name:
                    raise ValueError("相位列使用弧度，请先转换为度")
                raise ValueError("相位列标为 dB，单位冲突")
            columns.append(index)
        elif name in {"phase", "phaseangle"}:
            if require_degree and not allow_unqualified:
                raise ValueError("相位表头需要明确标出 deg 或 degrees")
            columns.append(index)
    if require_degree and len(columns) > 1:
        raise ValueError("表头中有多个角度列，请只保留一个")
    return columns


def read_curve(text, column=1):
    """Read one Hz/value curve, keeping the legacy default columns."""
    _, rows = _parse_table(text)
    return _curve(rows, 0, column)


def read_measurement(text, *, use_embedded_phase=True):
    """Read Hz/dB data and only use phase named explicitly in a table header."""
    header, rows = _parse_table(text)
    width = len(rows[0])
    if header is None:
        return _curve(rows, 0, 1), None

    tokens = [_header_token(label) for label in header]
    frequency_column, frequency_scale = _frequency_column(tokens, width)
    phase_columns = _phase_columns(tokens, require_degree=use_embedded_phase)
    phase_column = phase_columns[0] if use_embedded_phase and phase_columns else None
    excluded_columns = set(phase_columns)

    db_columns = [i for i, name in enumerate(tokens)
                  if i != frequency_column and i not in excluded_columns
                  and ("db" in name or "decibel" in name)]
    if len(db_columns) > 1:
        raise ValueError("表头中有多个 dB 列，请先整理成一条频响曲线")
    if db_columns:
        db_column = db_columns[0]
    else:
        unassigned = [i for i in range(width)
                      if i != frequency_column and i not in excluded_columns]
        if (len(unassigned) == 1 and width == 2
                and any(tag in tokens[unassigned[0]]
                        for tag in ("phase", "coherence", "quality", "radian", "degree", "deg"))):
            raise ValueError("该列未标明为 dB 频响")
        if len(unassigned) == 1:
            db_column = unassigned[0]
        elif width == 2:
            db_column = 1
        else:
            raise ValueError("请在表头标出唯一的 dB 频响列")

    if db_column in excluded_columns:
        raise ValueError("同一列不能同时作为 dB 频响和相位")
    measurement = _curve(rows, frequency_column, db_column, frequency_scale)
    phase = (_curve(rows, frequency_column, phase_column, frequency_scale)
             if use_embedded_phase and phase_column is not None else None)
    return measurement, phase


def read_phase(text):
    """Read a separate phase curve with a declared degree unit."""
    header, rows = _parse_table(text)
    width = len(rows[0])
    if header is None:
        return _curve(rows, 0, 1)

    tokens = [_header_token(label) for label in header]
    frequency_column, frequency_scale = _frequency_column(tokens, width)
    phase_columns = _phase_columns(tokens, allow_unqualified=True)
    if not phase_columns:
        if any("rad" in name for name in tokens):
            raise ValueError("相位列使用弧度，请先转换为度")
        if any("db" in name or "decibel" in name for name in tokens):
            raise ValueError("相位文件中找到 dB 列，请检查是否误选了频响文件")
        raise ValueError("相位文件表头需要标出 Phase 或角度单位")
    phase_column = phase_columns[0]
    if phase_column == frequency_column:
        raise ValueError("同一列不能同时作为频率和相位")
    return _curve(rows, frequency_column, phase_column, frequency_scale)


def read_inputs(measurement_text, phase_text=None):
    """Prefer a separately supplied phase file over a labeled embedded phase."""
    has_separate_phase = phase_text is not None
    measurement, embedded_phase = read_measurement(
        measurement_text, use_embedded_phase=not has_separate_phase
    )
    if has_separate_phase:
        return measurement, read_phase(phase_text), "separate"
    if embedded_phase is not None:
        return measurement, embedded_phase, "combined"
    return measurement, None, None


def log_interpolate(curve, frequency):
    hz, value = curve
    return np.interp(np.log(np.maximum(frequency, hz[0])), np.log(hz), value)


def minphase_spectrum(db):
    cepstrum = np.fft.irfft(db * np.log(10) / 20)
    cepstrum[1:len(cepstrum)//2] *= 2
    cepstrum[len(cepstrum)//2+1:] = 0
    return np.exp(np.fft.rfft(cepstrum))


def design(measurement, phase=None, rate=96000, *, demo=False):
    if rate not in (44100, 48000, 96000):
        raise ValueError("采样率须为 44100、48000 或 96000")
    nfft, taps = 65536, 8193
    reference = Path(__file__).with_name("p60_reference.csv")
    if demo:
        reference = reference.with_name("p60_dsp.csv")
    elif not reference.is_file():
        raise ValueError("请按 README 初始化固定 P60 目标。合成示例可直接运行。")
    ref_hz, ref_db, base_real, base_imag, wide_real, wide_imag = np.loadtxt(
        reference, delimiter=",", skiprows=1, unpack=True
    )
    target = (ref_hz, np.zeros_like(ref_db) if demo else ref_db)
    frequency = np.fft.rfftfreq(nfft, 1/rate)
    safe_frequency = np.maximum(frequency, 1)
    anchor = 500.
    curves = (measurement, target)
    lo = max(20, *(curve[0][0] for curve in curves))
    hi = min(14000, rate*.44, *(curve[0][-1] for curve in curves))
    if not lo < anchor < hi or hi / lo < 4:
        raise ValueError("曲线需有覆盖 500 Hz 且至少两倍频程的共同频段")
    logf = np.log2(safe_frequency)
    def support(lo, hi):
        left = np.clip((logf-np.log2(lo))/.5, 0, 1)
        right = np.clip((np.log2(hi)-logf)/.5, 0, 1)
        return np.sin(np.pi/2*np.minimum(left,right))**2

    measured_db = log_interpolate(measurement, frequency)
    target_db = log_interpolate(target, frequency)
    at = lambda values: float(np.interp(anchor, frequency, values))
    measured_db -= at(measured_db)
    target_db -= at(target_db)
    eq_db = np.clip(target_db-measured_db, -12, 6)*support(lo,hi)
    # Zero-phase magnitude EQ, realized below as a delayed finite FIR.
    common = 10**(eq_db/20) + 0j
    if phase is not None:
        # Input is measured acoustic phase in degrees, on a consistent timing reference.
        # A fixed measurement delay produces phase proportional to Hz.
        # Interpolating phase on a logarithmic axis would turn it into curvature.
        measured_phase = np.interp(frequency, phase[0], np.unwrap(np.deg2rad(phase[1])))
        remainder = measured_phase - np.unwrap(np.angle(minphase_spectrum(measured_db)))
        phase_lo = max(lo, phase[0][0])
        phase_hi = min(hi, phase[0][-1])
        if phase_lo > 300 or phase_hi < 1000:
            raise ValueError("相位校正需要共同覆盖 300–1000 Hz，才能估算测量时延")
        fit = (frequency >= 300) & (frequency <= 1000) & (frequency >= lo) & (frequency <= hi)
        slope, offset = np.polyfit(frequency[fit], remainder[fit], 1)
        inverse = -np.clip(remainder - (slope*frequency+offset), -np.pi/3, np.pi/3)
        common *= np.exp(1j*inverse*support(max(phase_lo,200),min(phase_hi,10000)))

    center = taps//2
    def truncate(spectrum):
        impulse = np.fft.fftshift(np.fft.irfft(spectrum, nfft))
        return impulse[nfft//2-center:nfft//2+center+1]*np.kaiser(taps, 5)

    def side_path(real, imag):
        ratio = log_interpolate((ref_hz, real), frequency)
        ratio = ratio + 1j*log_interpolate((ref_hz, imag), frequency)
        return truncate(common*(1+(ratio-1)*support(lo,hi)))

    mid = truncate(common)
    base = side_path(base_real,base_imag)
    wide = side_path(wide_real,wide_imag)
    sides = {"": (base+wide)/2, "base_": base, "wide_": wide}
    pairs = {prefix: ((mid+side)/2,(mid-side)/2) for prefix,side in sides.items()}

    def bound(pair):
        return float(sum(np.abs(x).sum(dtype=np.float64) for x in pair))

    # A sample-wise convex blend of the two capped stereo pairs cannot
    # exceed their maximum bound, even with a time-varying blend weight.
    kernel_bound = max(map(bound,pairs.values()))
    preamp_db = float(min(0, np.floor((-20*np.log10(kernel_bound)-1)*10)/10))
    gain = 10**(preamp_db/20)
    filters = {"eq_only": mid*gain}
    for prefix,(direct,cross) in pairs.items():
        filters[prefix+"direct"] = direct*gain
        filters[prefix+"cross"] = cross*gain
    margins = []
    for prefix in pairs:
        pair = [filters[prefix+k] for k in ("direct","cross")]
        rendered_bound = max(bound(pair),bound([x.astype(np.float32) for x in pair]))
        margins.append(-20*np.log10(rendered_bound))
    margin = float(min(margins))
    assert margin >= .99
    realized = 20*np.log10(np.maximum(abs(np.fft.rfft(mid,nfft)),1e-12))
    selection = np.geomspace(lo,hi,256)
    preview = np.column_stack([selection, *(np.interp(selection,frequency,x)
                              for x in (measured_db,target_db,measured_db+realized))])
    return dict(rate=rate, target="Synthetic demo" if demo else "P50 P60", spatial="WideAdaptive" if rate==96000 else "P60 fixed 50% blend", phase_used=phase is not None, preamp_db=preamp_db,
                sample_peak_margin_db=margin, latency_ms=1000*center/rate,
                preview=preview, filters=filters)


def playback_config(rate):
    """JSON is a YAML subset accepted by CamillaDSP."""
    def mixer(inputs, outputs, sources):
        return {"channels": {"in": inputs, "out": outputs}, "mapping": [
            {"dest": dst, "sources": [{"channel": src, "gain": 0.0} for src in channels]}
            for dst, channels in enumerate(sources)
        ]}
    return {
        "devices": {
            "samplerate": rate, "chunksize": 1024,
            "capture": {"type": "RawFile", "channels": 2, "format": "F32_LE",
                        "filename": "input.raw"},
            "playback": {"type": "File", "channels": 2, "format": "F32_LE",
                         "filename": "output.raw"},
        },
        "mixers": {
            "split": mixer(2, 4, [(0,), (1,), (1,), (0,)]),
            "sum": mixer(4, 2, [(0, 2), (1, 3)]),
        },
        "filters": {name: {"type": "Conv", "parameters": {
            "type": "Raw", "format": "TEXT", "filename": name + ".txt",
        }} for name in ("direct", "cross")},
        "pipeline": [
            {"type": "Mixer", "name": "split"},
            {"type": "Filter", "channels": [0, 1], "names": ["direct"]},
            {"type": "Filter", "channels": [2, 3], "names": ["cross"]},
            {"type": "Mixer", "name": "sum"},
        ],
    }


def adaptive_config(rate):
    if rate != 96000:
        raise ValueError("WideAdaptive requires 96 kHz")
    cfg = playback_config(rate)
    cfg["filters"] = {}
    cfg["mixers"]["split"] = {
        "channels": {"in": 2, "out": 8},
        "mapping": [{"dest": dst, "sources": [{"channel": src, "gain": 0.0}]}
                    for dst,src in enumerate((0,1,1,0,0,1,1,0))],
    }
    cfg["mixers"]["sum"] = {
        "channels": {"in": 8, "out": 4},
        "mapping": [{"dest": dst, "sources": [{"channel": src,"gain": 0.0}
                                            for src in channels]}
                    for dst,channels in enumerate(((0,2),(1,3),(4,6),(5,7)))],
    }
    cfg["mixers"]["take_output"] = {
        "channels": {"in": 4, "out": 2},
        "mapping": [{"dest": i,"sources": [{"channel": i,"gain": 0.0}]} for i in range(2)],
    }
    pipeline = [{"type": "Mixer","name": "split"}]
    for start,name in enumerate(("base_direct","base_cross","wide_direct","wide_cross")):
        cfg["filters"][name] = {"type": "Conv","parameters": {
            "type": "Raw","format": "TEXT","filename": name+".txt"}}
        pipeline.append({"type": "Filter","channels": [2*start,2*start+1],"names": [name]})
    cfg["processors"] = {"blend": {"type": "WideAdaptive"}}
    cfg["pipeline"] = pipeline + [
        {"type": "Mixer", "name": "sum"},
        {"type": "Processor", "name": "blend"},
        {"type": "Mixer", "name": "take_output"},
    ]
    return cfg


def realtime_config(backend, capture, playback):
    """Load the same 96 kHz adaptive matrix into the selected audio backend."""
    capture, playback = capture.strip(), playback.strip()
    if not capture or not playback or capture.casefold() == playback.casefold():
        raise ValueError("Select distinct capture and playback devices by name.")
    if playback.casefold().startswith(("blackhole", "cable input", "cable output")):
        raise ValueError("Select a physical playback device to avoid an audio feedback loop.")
    if backend not in {"CoreAudio", "Wasapi"}:
        raise ValueError("Unsupported audio backend")
    cfg = adaptive_config(96000)
    devices = cfg["devices"]
    devices.update(
        capture={"type": backend, "channels": 2, "device": capture},
        playback={"type": backend, "channels": 2, "device": playback, "exclusive": False},
        enable_rate_adjust=True,
        target_level=4096,
        volume_limit=0,
        stop_on_rate_change=True,
    )
    if backend == "Wasapi":
        devices["capture"].update(format="F32", exclusive=False, loopback=False)
        devices["playback"]["format"] = "F32"
        devices["resampler"] = {"type": "AsyncSinc", "profile": "Balanced"}
    else:
        if not capture.casefold().startswith("blackhole"):
            devices["resampler"] = {"type": "AsyncSinc", "profile": "Balanced"}
    return cfg


def eq_response(result):
    """Response of the exported common TEXT FIR, relative to its center tap."""
    taps = result["filters"]["eq_only"]
    rate = result["rate"]
    hz = result["preview"][:, 0]
    center = (len(taps) - 1) // 2
    _, transfer = freqz(taps, worN=hz, fs=rate)
    _, moment = freqz((np.arange(len(taps)) - center) * taps, worN=hz, fs=rate)
    magnitude = np.abs(transfer)
    valid = magnitude > 1e-12 * np.sum(np.abs(taps))
    with np.errstate(divide="ignore", invalid="ignore"):
        gain_db = 20 * np.log10(magnitude)
        phase_deg = np.angle(transfer * np.exp(2j * np.pi * hz * center / rate), deg=True)
        # Re{sum((n - center) h[n] exp(-jwn)) / H} is delay in samples.
        delay_ms = np.real(moment / transfer) * 1000 / rate
    return np.column_stack((hz, gain_db, np.where(valid, phase_deg, np.nan),
                            np.where(valid, delay_ms, np.nan)))


def bundle(result, live=None):
    dest = io.BytesIO()
    rate = result["rate"]
    if live is not None and rate != 96000:
        raise ValueError("Realtime adaptive playback requires 96 kHz.")
    with zipfile.ZipFile(dest,"w",zipfile.ZIP_DEFLATED) as archive:
        for name, values in result["filters"].items():
            buffer = io.BytesIO()
            wavfile.write(buffer, rate, values.astype(np.float32))
            archive.writestr(name+".wav",buffer.getvalue())
            archive.writestr(name+".txt","\n".join(f"{v:.17g}" for v in values)+"\n")
        archive.writestr("camilladsp-offline.yml", json.dumps(playback_config(rate), indent=2))
        if rate==96000:
            archive.writestr("camilladsp-adaptive.yml",json.dumps(adaptive_config(rate),indent=2))
        if live is not None:
            archive.writestr("camilladsp-realtime.yml",json.dumps(live,indent=2))
            archive.writestr("play.py",Path(__file__).with_name("play.py").read_text(encoding="utf-8"))
        report = {k: v for k, v in result.items() if k not in ("filters", "preview")}
        center = (len(result["filters"]["eq_only"]) - 1) // 2
        report["fir_center_samples"] = center
        if rate == 96000:
            # LEAD in the 96 kHz WideAdaptive processor.
            report["adaptive_lookahead_samples"] = 672
            report["adaptive_reference_delay_ms"] = (center + 672) * 1000 / rate
        archive.writestr("report.json", json.dumps(report, indent=2))
        data = io.StringIO()
        np.savetxt(data,result["preview"],delimiter=",",
                   header="hz,measured_db,target_db,corrected_mono_db",comments="")
        archive.writestr("response.csv",data.getvalue())
        response = io.StringIO()
        np.savetxt(response, eq_response(result), delimiter=",", comments="",
                   header="hz,eq_with_preamp_db,eq_centered_phase_deg,eq_centered_group_delay_ms")
        archive.writestr("eq-response.csv", response.getvalue())
        archive.writestr("README.txt", "Wet L = direct*L + cross*R; wet R = direct*R + cross*L.\n"
                         "The EQ-only file is an alternative. * denotes convolution. All filters include preamp.\n"
                         "Use camilladsp-adaptive.yml at 96 kHz with the patched engine.\n"
                         "Rates must match. The peak bound covers digital samples at the FIR output only.\n")
    return dest.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurement",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--phase",type=Path, help="Separate Hz/degrees file with known acoustic time reference")
    parser.add_argument("--rate",type=int,default=96000)
    parser.add_argument("--backend", choices=["CoreAudio","Wasapi"])
    parser.add_argument("--capture", default="")
    parser.add_argument("--playback", default="")
    parser.add_argument("--demo", action="store_true", help="Use a synthetic demo target")
    args = parser.parse_args()
    if (args.capture or args.playback) and not args.backend:
        parser.error("--capture and --playback require --backend")
    if args.backend and args.rate != 96000:
        parser.error("Realtime adaptive playback requires 96 kHz")
    phase_text = args.phase.read_text(encoding="utf-8-sig") if args.phase else None
    measurement, phase, phase_source = read_inputs(
        args.measurement.read_text(encoding="utf-8-sig"), phase_text
    )
    result = design(measurement, phase, args.rate, demo=args.demo)
    result["phase_source"] = phase_source
    live = realtime_config(args.backend,args.capture,args.playback) if args.backend else None
    args.output.write_bytes(bundle(result,live))


if __name__ == "__main__":
    main()
