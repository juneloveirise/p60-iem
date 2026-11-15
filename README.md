# P60 IEM 校正与声场

在本机导入耳机频响和可选声学相位，以 P50 P60 的居中等效频响为固定目标，预览校正曲线并导出 44.1、48 或 96 kHz 的单通道校正 FIR 和对称立体声 FIR。测量数据在本机处理。

本项目由 **OpenAI Codex（AI）** 在项目发起人的需求、调音取舍与试听反馈下编写、移植和测试。项目独立于 EPZ、HuiHiFi 与 CamillaDSP 上游。

## 启动

安装 Python 3.12。在本目录运行：

```text
python -m venv .venv
```

| 系统 | 安装 | 启动 |
| --- | --- | --- |
| macOS | `.venv/bin/python -m pip install -r requirements.txt` | `.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false` |
| Windows | `.venv\Scripts\python.exe -m pip install -r requirements.txt` | `.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false` |

终端也可运行 `python iem.py examples/measurement.csv filters.zip --phase examples/phase.csv --demo`。

## 首次准备固定 P60 目标

合成演示可直接运行。正式使用前，取得 [HuiHiFi EPZ P50 原始频响](https://huihifi.com/evaluation/185a3d4a-fc5b-458b-b0a5-db4bb716d7cf)的 Hz/dB 文本或 RAW_FR JSON，在项目目录运行 `python prepare_reference.py <P50原始频响文件>`。

程序把固定的 P60 数字传递与 P50 原始频响合成，生成本地参考。此后每副耳机只需导入自己的测量。源码包附数字滤波传递和合成示例，HuiHiFi 原始测量由使用者自行取得。

## 输入和输出

- UTF-8 CSV、TSV 或空白分隔的文本。每行先写 Hz，再写 dB 或度。支持普通表头。组合文件中的相位需另存为 Hz 和度两列。频点须唯一。
- 耳机测量与目标需要可比较的测量基准。相位应来自同次声学测量，有可信的时间参考，且频点足够密以重建展开相位。导入相位后，在 300–1000 Hz 拟合并去掉线性延时及常量。实测相位仍会受测量和佩戴影响。
- 频响按 500 Hz 对齐，限制校正幅度，频带两端渐退。当前使用一条共同测量曲线处理两耳。生成的幅度校正采用线性相位 FIR。两套声场滤波使用 P60 的基础与加宽复数侧/中传递比。96 kHz 导出实时自适应混合，44.1 和 48 kHz 导出固定 50% 混合。
- ZIP 含 WAV、TEXT、频响 CSV、预衰减与数字样本余量报告，以及标准 CamillaDSP 文件输入输出配置。在解压目录运行 `camilladsp camilladsp-offline.yml`，输入为所选采样率的双声道交错 float32 little-endian `input.raw`，输出为 `output.raw`。配置的输入输出由用户另行接到播放设备。FIR 已计入预衰减。
- 数字余量按每路卷积系数绝对值和计算，覆盖 FIR 输出的样本峰值。模拟输出和采样点之间的峰值需另行验证。

96 kHz 的 `camilladsp-adaptive.yml` 需要加入 WideAdaptive 的 CamillaDSP。该处理器按帧混合基础和加宽声场，原来的 P60 检测时间和权重规则保持一致。它读取目标耳机的校正后信号，两套滤波的输出按同一个权重混合。预衰减按照两套输出的较坏数字样本上界计算。

在 CamillaDSP v4.1.3 源码目录应用补丁并编译。需要 Git 和 Rust：

```text
git clone --branch v4.1.3 --depth 1 https://github.com/HEnquist/camilladsp.git upstream
git -C upstream apply --check <补丁绝对路径>
git -C upstream apply <补丁绝对路径>
cargo build --release --locked --manifest-path upstream/Cargo.toml
```

补丁路径由 Git 进程解析，请从工具目录传入 `engine/wide-adaptive.patch` 的绝对路径。解压导出包后，在解压目录用编译好的程序运行 `camilladsp-adaptive.yml`。标准引擎仍可用 `camilladsp-offline.yml` 渲染静态对照。

## 实时播放

首次把系统或播放器的输出设成虚拟设备。随后在工具的「实时播放设备」内选音频系统，填采集端和真正的耳机／DAC 输出设备名，再点「一键生成」。每副耳机只需导入它的频响和可选声学相位，生成器会自动套用 P60 目标。

| 系统 | 播放器输出 | 工具里的采集端 | 时钟 |
| --- | --- | --- | --- |
| macOS | BlackHole 2ch 或已安装的 BlackHole 设备 | 同名 BlackHole | 利用 BlackHole 虚拟时钟同步硬件 |
| Windows | CABLE Input (VB-Audio Virtual Cable) | CABLE Output (VB-Audio Virtual Cable) | WASAPI 共享模式，用 AsyncSinc 跟踪时钟漂移 |

Mac 在「音频 MIDI 设置」中将虚拟设备和物理输出设为 96 kHz，首次运行按系统提示授予采集权限。Windows 在声音控制面板中把 CABLE Input、CABLE Output 和物理输出的默认格式设到 96 kHz，设备名以系统实际显示为准。若设备没有 96 kHz 档位，可以用较低采样率导出静态滤波。

解压生成包，在包内运行 `python play.py <编译后的 CamillaDSP 程序路径>`。CamillaDSP 在前台输出设备连接状态。按 Ctrl+C 停止。重新生成另一副耳机时，先退出这一实例，再从新的导出目录启动。

本地验收覆盖 Mac Core Audio 静音启动和离线自适应渲染。Windows 的 WASAPI 设备连接、长时稳定性和听感仍待在 Windows 主机验证，构建和配置校验已纳入随包的双系统 CI。

固定目标取自 Hui P50 实测和已载入的 P60 数字链模型，包含 3.776 kHz 补偿。P50 内置八段 EQ 的逆相位滤波不跨设备复制。导入声学相位时，程序只对输入测量的剩余相位做限带、限幅修正。该处理不保证跨耳机的实际双耳声学相位或主观听感相同。

项目源码使用 MPL 2.0。CamillaDSP 补丁随附上游 GPLv3 和 MPL 2.0 许可证。
