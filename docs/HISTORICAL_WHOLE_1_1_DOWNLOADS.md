# 物欲完整体 1.1：交付与复现入口

2026-10-12。已完成主干实现、1,024 条正式生命、全量物理审计与独立解压验证。

结论：工程接通完成；动作学习改善稳定环境的身体维持，持续状态候选在复杂挑战中的优势尚未确立。默认保持修正后的 A。

## 先看报告

[WUYU_Whole_Report_2026-10-12.html](https://chatgpt.com/api/library/files/libfile_a53e60e834748191a19abd29f297c275/download)（独立 HTML，图表已内嵌）

[WUYU_Whole_Report_2026-10-12.md](https://chatgpt.com/api/library/files/libfile_f2b84ab6a268819194a69e5eecb0443a/download)

## 运行源码

[WUYU_Whole_Source_2026-10-12.zip](https://chatgpt.com/api/library/files/libfile_fd9dcc4c937c8191869a018514a89575/download)（268.0 MB）

Source 包包含物欲 1–8、后续研究的全部既有源码与本轮新主干、所需既有模型、机制说明、迁移工具、检查点和结论。仅下载 Source 即可运行默认入口、开启 A/B/C/D、保存及继续个体。

## 本轮全部原始数据

将 Source 与以下六个 Data 包解压到同一个父目录。它们共享 `WUYU_Whole_2026-10-12/` 顶层目录，文件不重叠；六个数据包合计约 1.54 GB。

| 文件 | 大小 |
|---|---:|
| [WUYU_Whole_Data_01_2026-10-12.zip](https://chatgpt.com/api/library/files/libfile_475e6801fbf08191a02d041967ca4bbb/download) | 277.1 MB |
| [WUYU_Whole_Data_02_2026-10-12.zip](https://chatgpt.com/api/library/files/libfile_578b47b019ac8191a7d964f9f5d1f7de/download) | 276.2 MB |
| [WUYU_Whole_Data_03_2026-10-12.zip](https://chatgpt.com/api/library/files/libfile_ec39be23fca081919f843c93df49d6fd/download) | 276.5 MB |
| [WUYU_Whole_Data_04_2026-10-12.zip](https://chatgpt.com/api/library/files/libfile_d876a87456708191beab6ea628238689/download) | 276.0 MB |
| [WUYU_Whole_Data_05_2026-10-12.zip](https://chatgpt.com/api/library/files/libfile_761d48bb96e881918980492095dfd896/download) | 276.1 MB |
| [WUYU_Whole_Data_06_2026-10-12.zip](https://chatgpt.com/api/library/files/libfile_50fab7f8ca788191a0e2e8eebf5f68c6/download) | 157.1 MB |

数据包含两轮开发、独立语义回归、正式 holdout 与状态诊断原始记录。失败个体、负结果及两份逐字节恢复的轨迹记录均保留。历史大体积研究数据仍使用上一轮七个历史 Data 包；不要与这里六个新 Data 包混淆。

## 接续与核验

[WUYU_Whole_HANDOFF_2026-10-12.md](https://chatgpt.com/api/library/files/libfile_2bba6aacc5648191acae086c75c7f65a/download)

[WUYU_Whole_SHA256SUMS_2026-10-12.txt](https://chatgpt.com/api/library/files/libfile_0968ecf1c4088191a9b8345549c77c91/download)

[WUYU_Whole_Package_Verification_2026-10-12.json](https://chatgpt.com/api/library/files/libfile_850b967b51608191b3ed428cef16e545/download)

[WUYU_Whole_Archive_Manifest_2026-10-12.json](https://chatgpt.com/api/library/files/libfile_2e9cb01ece648191882636fd1e717d53/download)

七个压缩包共 6,210 个成员逐文件 SHA256 一致且无重叠。独立解压后，源码/协议 300 项检查通过；第 73 步暂停的跨步学习状态可精确恢复到第 136 步，正式检查点后续轨迹也一致。

## 实际入口

在解压后的项目目录运行：

```bash
python3 -m pip install -r requirements.txt
python3 -B -m wuyu_unified run --seed 990001 --steps 384 --out runs/whole_384
python3 -B -m wuyu_unified resume runs/whole_384/state.wuyu --steps 384 --out runs/whole_768
```

默认是 `whole/A/native`。要运行持续状态候选，显式使用 `--mode D --case challenge --horizon 2560`。完整复现实验命令在 README，固定方案在 `experiments/whole_holdout/protocol.json`。
