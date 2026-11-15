"""Local browser interface for the FIR generator."""
from pathlib import Path
import platform

import numpy as np
import streamlit as st

from iem import bundle, design, read_curve, realtime_config

st.set_page_config(page_title="IEM 校正与声场", page_icon="🎧")
st.title("IEM 校正与声场")
st.caption("固定目标 · P50 · P60")
demo = st.checkbox("使用合成示例", value=True)
if demo:
    folder = Path(__file__).parent / "examples"
    measurement = (folder / "measurement.csv").read_text()
    phase_text = (folder / "phase.csv").read_text()
    st.caption("合成演示使用平直目标，便于检查导入和导出。")
else:
    def upload(label, key):
        file = st.file_uploader(label, type=["csv", "txt", "tsv"], key=key)
        return file.getvalue().decode("utf-8-sig") if file else None

    try:
        measurement = upload("耳机频响 · Hz / dB", "measurement")
        phase_text = upload("声学相位（可选）· Hz / 度", "phase")
    except UnicodeDecodeError:
        st.error("请用 UTF-8 保存测量文件。")
        st.stop()

rate = st.selectbox("导出采样率 (Hz)", [96000, 48000, 44100])
st.caption("96 kHz 导出实时自适应配置。44.1 / 48 kHz 导出固定 FIR。")
if not measurement:
    st.info("导入耳机频响即可生成。声学相位文件可选，需有可信时间参考。")
    st.stop()
try:
    result = design(read_curve(measurement), read_curve(phase_text) if phase_text else None, rate, demo=demo)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

import pandas as pd
frame = pd.DataFrame(result["preview"], columns=["Hz", "测量", "目标", "校正后"])
frame["频率（对数）"] = np.log10(frame["Hz"])
st.line_chart(frame, x="频率（对数）", y=["测量", "目标", "校正后"])
a, b, c = st.columns(3)
a.metric("已计入的预衰减", f'{result["preamp_db"]:.1f} dB')
b.metric("数字样本余量", f'{result["sample_peak_margin_db"]:.1f} dB')
c.metric("FIR 参考延时", f'{result["latency_ms"]:.1f} ms')
live = None
if rate == 96000:
    with st.expander("实时播放设备 · 首次设置"):
        is_windows = platform.system() == "Windows"
        backend = st.selectbox("音频系统", ["CoreAudio", "Wasapi"], index=int(is_windows))
        capture_name = {"CoreAudio": "BlackHole 2ch",
                        "Wasapi": "CABLE Output (VB-Audio Virtual Cable)"}
        capture = st.text_input("采集设备", value=capture_name[backend], key=f"capture-{backend}")
        playback = st.text_input("耳机或 DAC 的系统设备名", key=f"playback-{backend}")
        if not playback:
            st.caption("填入输出设备名后，下载包会附上实时播放配置。")
        if playback:
            try:
                live = realtime_config(backend,capture,playback)
            except ValueError as exc:
                st.error(str(exc))
                st.stop()
            st.caption("导出后解压，运行 python play.py 跟上 CamillaDSP 程序路径。")
st.download_button("一键生成", bundle(result,live), "p60-filters.zip", "application/zip")
