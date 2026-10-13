# WUYU 源码检索与完整性核对

核对日期：2026-10-12（Asia/Shanghai）。本次任务：仔细阅读交接材料，并寻找资料库中现有的全量源码。

已读完所附交接报告和理论底稿全文，阅读当前实验的说明、方案、报告、核心机制与运行器；取回并检查 18 个相关压缩包。对旧大工程完成归档清点与比对，没有声称逐行理解约五十万行历史 Python 源码。

## 结论

01—08 的原始完整包均已找到，统一工程和当前目标形成实验也已找到。整合包覆盖 01—08 的全部 115 个 Python 源文件，内容逐文件一致；连同整合层的 18 个 Python 文件，合计 133 个。当前目标形成实验为独立项目，另有 6 个 Python 文件。

但不能把 WUYU_Integrated_Full.zip 当作资料库全部历史材料的全集：它对 06—08 使用的是 run 包，缺少这些阶段 complete 包中的 628 个成员，包括模型、检查点、原始数据、配置、日志和一个报告 JavaScript 文件。三个 complete 包本次均已取回。

当前资料库未找到可核验的独立 0 阶段原始源码包；不能据此宣布“0—8 全齐”，也不能将其他早期工程自行认定为 0。

## 接手时保持的理解

- 目标是研究由物理欲望形成的数字个体，涉及理解、关系、历史、价值与自身改变；旧代码作为参考，不限定最终架构。
- 当前理论是开放的候选形成理论；第一轮目标形成实验只检验给定调节准则下的工具性选择、换路、重估和停止。
- 删除显式目标缓冲并不降低当前实验表现；不能把持续行为直接当作持续欲望成立的证据。
- 有限自主探索是交接建议，当前归档没有实现或运行该后续实验；本次没有新写物欲机制、重训模型或启动后续实验。

## 应从哪些文件开始

| 用途 | 准确文件名 | 内容与边界 |
|---|---|---|
| 当前理论 | WUYU_Theory_Working_Draft_2026-10-11.md | 当前取回内容与恢复包中的版本完全一致 |
| 当前局部实验 | WUYU_Goal_Formation_Study.zip | 独立表格学习与图搜索项目；源码、方案、报告和原始结果齐备 |
| 旧统一工程日常阅读、运行 | WUYU_Integrated_Run.zip | 133 个 Python 文件；与 Full 的 Python 源码完全一致 |
| 旧统一工程及较多历史产物 | WUYU_Integrated_Full.zip | 01—08 原始 Python 源码与统一运行层；仍需三个后期 complete 包补齐历史产物 |
| 各阶段最完整独立交付 | WUYU_01_complete.zip 至 WUYU_08_complete.zip | 保留每个阶段自身的配置、源码、报告和实验材料 |
| 早期大工程参考 | laap-AGI--feat-v4.0-full-stack (1).zip | 2,081 个 Python 文件、498,415 行；另含前端等文件 |
| 早期另一组实现 | 源码.zip 或 源码(1).zip | 两包字节相同；55 个 Python 文件、23,197 行；不是大工程的简单同内容子集 |
| 历史审阅 | WUYU_RESEARCH_REVIEW.zip | 审阅报告、静态索引和少量用于探查的源码，不是历史源码全集 |

文件名用于后续准确检索。原压缩包未修改。当前工作目录为 /workspace/scratch/0da092536cba/sources；该路径只是本次取回位置，跨会话应按准确文件名重新获取。

## 01—08 覆盖核对

| 阶段 | 原包 Python 文件数 | 与整合包对应文件一致数 | 核心入口（在各阶段目录下） |
|---|---:|---:|---|
| 01 | 7 | 7 | wuyu/core.py |
| 02 | 14 | 14 | wuyu02/workshop.py |
| 03 | 15 | 15 | wuyu03/core.py、neural.py、control.py |
| 04 | 10 | 10 | wuyu04/latent.py、diagnostic.py |
| 05 | 10 | 10 | wuyu05/core.py |
| 06 | 21 | 21 | wuyu06/agent.py、world.py |
| 07 | 20 | 20 | wuyu07/experiment.py |
| 08 | 18 | 18 | wuyu08/experiment.py、network.py |

这些数字按包内 .py 文件计数，包含测试、辅助脚本和包中保留的历史副本，不是独立机制数。

统一工程的入口为 WUYU_INTEGRATED/wuyu_core/__main__.py；适配器 wuyu_core/legacy.py 只注册 01—08。原版分别位于 legacy/WUYU_01 至 legacy/WUYU_08。

当前实验入口为 WUYU_Goal_Formation_Study/wuyu_goal_study/core.py 和 run_study.py。这里的 Learner.decide 在空目标目录或没有已观察的可达结果时返回停止；所以不能直接当成自主探索实现。

## 完整性差别与来源冲突

### Full 与 Run

两者包含完全相同的 133 个 Python 文件。Full 总计 1,114 个文件，Run 总计 924 个；共有 923 个成员字节相同，DISTRIBUTION.json 因打包种类而不同，Full 另有 190 个成员。

### 整合包与独立 complete 包

| 来源 | 整合包未包含的原成员数 | 内容 |
|---|---:|---|
| 01、02、04、05 | 0 | 原成员齐全且一致 |
| 03 | 1 | 可再生 .pyc 缓存；15 个 Python 源码文件全齐 |
| 06 | 123 | 111 个 .npz、5 个 .jsonl、4 个 .partial、3 个 .json |
| 07 | 103 | 103 个 .npz |
| 08 | 402 | 370 个 .npz、24 个 .log、7 个 .json、1 个 .js |

08 额外脚本的准确路径是 WUYU_08/engineering/report_script.js，42,303 字节。06—08 的 CHECKSUMS.sha256 因轻量包内容不同而不同；整合版 06 另有 LIGHT_BUNDLE.txt。这些差别不等于源码受损。

因此：研究 Python 实现时，整合包已覆盖现存 01—08；需要保留全部历史实验产物、工程试跑资料与独立脚本时，必须同时保留各阶段 complete 包。

### 早期两个“源码”包

源码.zip 与 源码(1).zip 的 SHA-256 完全相同。其 55 个 Python 文件与 LAAP 大包按相对路径比较：4 个同路径文件内容不同，51 个没有同路径对应项；没有同路径同内容文件。两类材料都需保留，不能直接覆盖合并。

### 0 阶段与旧 0A 标记

当前资料库全量目录没有独立 0/00/0A 原包，统一工程只包含 01—08。旧审阅包的 code_inventory.json 里残留 WUYU_0A 标签和 5 个文件的路径、哈希，但它不是实际源码；本次在 18 个取回压缩包的 Python 成员中均未找到这些哈希对应的文件内容。用户此前明确纠正“没有0A”，因此保留该条为旧记录冲突，不把它确认为存在过的合法阶段，也不伪造同名恢复代码。

## 本次核验范围

- 附件恢复包：26 个列入校验表的成员 SHA-256 全部一致；单独交接 Markdown 与包内版本一致。
- 取回的 18 个相关压缩包：所有成员均通过 ZIP CRC 检查。
- 统一工程 Full：1,113 个列入分发校验表的成员全部匹配；Run：923 个全部匹配。
- 整合包记录的 8 个来源压缩包 SHA-256 与本次取得的原包一致。
- 当前实验原始 ZIP 的 21 个成员与恢复包中对应项目文件全部一致；理论底稿也一致。
- 对 01—08 所有 Python 源码做逐文件哈希比较；另核对旧小源码包和大工程之间的重复与差异。
- 本次没有重跑历史训练、实验或全部单元测试；历史报告中的通过数量属于原报告记录，不冒充本次实测。

## 全部已取回压缩包清单

大小采用十进制 MB；Python 数为包内 .py 文件数，不做跨包去重。

| 文件名 | MB | 文件数 | Python 文件数 |
|---|---:|---:|---:|
| laap-AGI--feat-v4.0-full-stack (1).zip | 143.130 | 13801 | 2081 |
| WUYU_01_complete.zip | 6.178 | 39 | 7 |
| WUYU_02_complete.zip | 13.082 | 73 | 14 |
| WUYU_03_complete.zip | 65.531 | 65 | 15 |
| WUYU_04_complete.zip | 20.766 | 49 | 10 |
| WUYU_05_complete.zip | 89.546 | 68 | 10 |
| WUYU_06_complete.zip | 243.533 | 185 | 21 |
| WUYU_06_run.zip | 3.700 | 63 | 21 |
| WUYU_07_complete.zip | 31.183 | 200 | 20 |
| WUYU_07_run.zip | 1.459 | 97 | 20 |
| WUYU_08_complete.zip | 326.375 | 515 | 18 |
| WUYU_08_run.zip | 6.424 | 113 | 18 |
| WUYU_Goal_Formation_Study.zip | 0.854 | 21 | 6 |
| WUYU_Integrated_Full.zip | 236.417 | 1114 | 133 |
| WUYU_Integrated_Run.zip | 43.645 | 924 | 133 |
| WUYU_RESEARCH_REVIEW.zip | 0.514 | 13 | 5 |
| 源码(1).zip | 0.310 | 59 | 55 |
| 源码.zip | 0.310 | 59 | 55 |

## 压缩包 SHA-256

| 文件名 | SHA-256 |
|---|---|
| laap-AGI--feat-v4.0-full-stack (1).zip | d6a8a78e0a8109612e10ee89f4cb41faed609cf662da7eb3e15f1caea478bb92 |
| WUYU_01_complete.zip | c4976c347eadb5c371b14b6f5c0c891f1490e45cdf1acfae1bf3ca2c1af3c56e |
| WUYU_02_complete.zip | 05e615619cc0dced0852c5ccc6bbdc79192a1d245c3264966328f1346f7ce93b |
| WUYU_03_complete.zip | b47f64d5b644402a63529c322520c5fa18c74d4f608bd5d12d036f331532e778 |
| WUYU_04_complete.zip | 9f8533e2f6426c594ee54028c815a06efab88e72a4e1349ab5d020ea7b5ec4f3 |
| WUYU_05_complete.zip | ab09d9d1b18535f0770e04627243b1d72f5b1c11bc65633b5d17cc54aff58f45 |
| WUYU_06_complete.zip | b9a026a59d3a785777b9ac4f10346506d7b40120d05638cf29d36afe25cd799d |
| WUYU_06_run.zip | 19740b7a81119361f2dfe9894546e250645dd782e44b2c9d577053e992e50c2d |
| WUYU_07_complete.zip | e227d1d25724092817d492d70f61a18c842b64abf6a6ca9408176f4eb84979d1 |
| WUYU_07_run.zip | 6284b46fec540d94fc05f586a671b242ea43190c0a8f5247184c30b2a3418e08 |
| WUYU_08_complete.zip | 9c3190f3ae62b89947e523ff68c9c6988a39fd0618211278771eb2e681b80ef0 |
| WUYU_08_run.zip | 0281a317ca0cb1f26f17dba9100f7c211db8ff9fa87aa718f840cb309cfd95e6 |
| WUYU_Goal_Formation_Study.zip | f758222ef7f134c8f232257d70663d6977e50c0fdc30ed08e2f17ee0f4c45a5f |
| WUYU_Integrated_Full.zip | 361d6c6cc295050c9bbc914f2cf3aecd6130e176623e64a7c6a025e9dde289bd |
| WUYU_Integrated_Run.zip | cfa49e9248b451112ea170ec0552148e644f86e99ccac6c6a72824562d0987f7 |
| WUYU_RESEARCH_REVIEW.zip | cf0d0b8b478732edcc179255e745990c3d220b395dd633652d7066e73a6cd0da |
| 源码(1).zip | 670d25b1fae479361a09ae32eacb2d9b76df98dfdaeb9f49353d71a496db8814 |
| 源码.zip | 670d25b1fae479361a09ae32eacb2d9b76df98dfdaeb9f49353d71a496db8814 |

本清单记录本次可核验材料，不将已找到的源码等同于最终物欲已经完成。
