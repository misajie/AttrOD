# AttrOD 完整阶段规划（draft2 主文档 · EXPERIMENT_PLAN 闸门）

> 用途：给 clone / urbansem / Server / Codex Sync 共用。工程边界写进本文，减少口头解释。  
> 科学设定以 `docs/draft2.md` 为准；切片与闸门以 `docs/EXPERIMENT_PLAN.md` 为准。  
> **禁止额外发明 draft2 没有的科学设定。**

## 0. 总览

| 目标 | 内容 |
| --- | --- |
| 科学问题 | Condition S：各发布切分 vs 晨间 HBW 参考 `R`；Condition G：仅在 HBW `R` 上拟合生成器，再对目标切分做 production-constrained 生成并打分（CPC 等） |
| 数据主线 | MITMA（完整日栈）+ Lombardy；HK 仅 verification / 契约 smoke，**不得**升为论文主线证据 |
| 工作区 | **代码**：`F:\processing\AttrOD`（code-only）→ push → myserver `~/AttrOD`；**数据**：只在 myserver `~/AttrOD/data`，禁止进 Mac / `F:` |
| 座位 | urbansem Sync 写代码；Server Sync sync+跑；Codex Sync 规划/验收；clone bot 协助本地环境 |

### 硬禁令（全程）

1. 数据集不得落 Mac / `F:\processing`。  
2. `distance_status=provisional_wgs84_not_metric` 时：**不得宣称 paper mainline**（`ok_mainline=false`；若 readiness JSON 字段矛盾，以运行为准并修字段）。  
3. FID 缺失必须 quarantine（HK FID9）；未 quarantine 不得扩主线。  
4. Condition G 训练输入只能是 HBW `R`；输出必须过 `O_i^T` 约束。  
5. DeepGravity pin `8693536`；NeuroGravity pin `7e29ef0`；**stub 与正式 edge-enhanced 分列**，stub 不得标正式主结果。  
6. 单套 `build_run_manifest` / run ledger schema；失败保留 stage + input_hash + error。  
7. 距离最终必须是 **metric CRS**（draft2：ETRS89/UTM 等）；provisional WGS84 只能用于契约/接口验证。

---

## 1. 阶段总表

| 阶段 | 状态 | 目标（对照 draft2） | 进入下一阶段条件 |
| --- | --- | --- | --- |
| NOW-0 数据与运行契约 | ✅ | data_root、readiness、FID quarantine、距离状态入 manifest | 契约可审计 |
| NOW-1 HK Condition S smoke | ✅（有产物） | S：`λR`/`λRᵀ`/`Δ`、null、day bootstrap、partition | verification_only |
| NOW-2 HK Condition G 经典矩阵 | 🔄 NOW-5 落地中 | gravity(power/exp)+closed-form、radiation、RF、IPF、oracle + `O_i^T` QA | 八模型目录齐全且 ledger 可重放 |
| NOW-3 DG/NG adapter 审计 | ✅（stub/dry-run） | import、schema round-trip、provenance；NG=stub | 不进论文主表 |
| NOW-4 run ledger | ✅ | `attrOD-run` → `outputs/runs/<run_id>/` | 已验收 |
| NOW-5 draft2 路径接通 | 🔄 Server 跑 accept | smoke→S→G 经典矩阵挂 ledger + readiness 字段修复 | 样例 run + 八模型 + 字段正确 |
| **P1 距离去 provisional（metric CRS）** | ⏳ | 按 draft2 重算 `d_ij`/`d_ii`，关掉 provisional | `distance` 非 provisional → 允许谈主线 |
| **P2 MITMA 主线 Condition S** | ⏳ | draft2 S：全切分 vs `R`、partition、null、day bootstrap | 31/31 + metric 距离 + join 闭合 |
| **P3 MITMA 主线 Condition G 零样本** | ⏳ | 经典矩阵 + Deep Gravity（pin）；production-constrained | 训练仅 HBW `R`；约束 QA |
| **P4 neuroGravity few-shot** | ⏳ | draft2 few-shot：meta-Gravity + edge GT；mask 无泄漏 | stub/正式分列；正式版依赖方案已定 |
| **P5 Lombardy 主线 S+G** | ⏳ | 同协议；无 day-bootstrap（单日） | Lombardy/OSM download manifest+checksum |
| **P6 表图与可复现打包** | ⏳ | draft2 表/图所需源数据与 caption 字段 | 全部 run_id / seed / commit 可追溯 |

> 命名：NOW-* = provisional/HK 可跑切片；P* = 完整实验主路径（原 WHEN-1/2/3 映射到 draft2，不再只写 smoke）。

---

## 2. 分阶段说明（完整实验）

### NOW-0 … NOW-4（已完成要点）

- 数据与闸门契约在 myserver；HK smoke verification_only。  
- ledger：`outputs/runs/<run_id>/{run_manifest.json,status.json,logs/}`。  
- readiness：`daily_trips` 31/31 已清陈旧 `found_n=10`；距离仍 provisional → **运行 kill mainline**。

### NOW-5（当前票 · 不创新）

- **做**：按 draft2/计划已写内容，把 HK smoke→Condition S→Condition G **经典矩阵**挂进 `attrOD-run`；修 readiness：provisional ⇒ `paper_mainline_allowed=false` / `verification_only=true`。  
- **经典矩阵（至少）**：Gravity_power、Gravity_exp、Radiation、RandomForest、IPF（含计划中的 row 变体）、Oracle、ClosedFormGravity。  
- **不做**：新科学设定；DG/NG 正式主结果；MITMA/Lombardy 主线宣称。  
- **验收**：`attrOD-run --pipeline hk_smoke_classic ...` → run 目录 + `outputs/hk_smoke/condition_g/{model}/` 齐全；闸门字段正确。

### P1 · 距离 metric CRS（主线总开关）

- **目标**：实现 draft2 距离定义（质心 Euclidean + 区内 `√(A/π)`），写入 manifest（CRS、单位、异常值）。  
- **输出**：距离矩阵/缓存 + `distance_status=metric_…`。  
- **验收**：provisional 标记消失；主线配置可读到 metric 距离。  
- **未完成前**：禁止 P2–P6 的 paper mainline 产物。

### P2 · MITMA Condition S 主线

- **输入**：`data/mitma/daily_trips`（31 完整日）、districts、population；metric 距离。  
- **按 draft2**：构建 `R`（晨间 HBW）；对各发布切分建 `T`；`λR`/`λRᵀ`/independence null；有 ≥2 日则 day bootstrap。  
- **输出**：`outputs/mainline/mitma/condition_s/`（metrics、partition、null、bootstrap、QA、manifest）。  
- **验收**：与 HK smoke **同一契约**；可重放；无未解释 NaN/Inf。

### P3 · MITMA Condition G 零样本主线

- **模型**：draft2 零样本：gravity（power/exp/closed-form）、radiation、Deep Gravity（pin）；另保留计划中的 RF/IPF/oracle 作为工程矩阵若 draft2 表需要。  
- **规则**：只在 HBW `R` 上训练/拟合；推理后 `O_i^T` 约束。  
- **输出**：`outputs/mainline/mitma/condition_g/{model}/`。  
- **验收**：约束 QA；provenance（commit/seed/hash）；全部 verification 字段在未达论文门槛前保持诚实。

### P4 · neuroGravity few-shot（draft2）

- **设定**：meta-Gravity 先验 + edge-enhanced GT；declared fraction 的 target edges；CPC on held-out。  
- **闸门**：正式 edge-enhanced 前，stub 只能诊断；`torch_geometric`/官方集成方案未定不得标主结果。  
- **输出**：mask manifest、无泄漏审计、stub/正式分列指标。

### P5 · Lombardy（+ OSM 若 Deep Gravity 需要）

- **数据**：download manifest、checksum、来源时间戳；zones/population/OD/OSM join 闭合。  
- **协议**：与 MITMA 相同 S/G；**无** day-bootstrap。  
- **距离**：Monte Mario / Italy zone 等 draft2 指定 CRS——禁止沿用 HK provisional。

### P6 · 表图与交付

- 按 draft2 表/图清单导出源数据与统计；每个对象绑定 `run_id`、seed、git commit、data hash。  
- 不新增城市/Table 1 条目除非 draft2 已要求。

---

## 3. 推荐排票顺序（立即）

1. **收口 NOW-5**：Server 确认 `f8b27e1` accept 日志与八模型目录 + readiness 字段。Codex Sync 验收。  
2. **P1 metric CRS**（下一正式票）：解锁主线的唯一硬开关。  
3. **P2 → P3** MITMA S 然后 G 零样本。  
4. **P4** few-shot（正式 NG 方案并行决策）。  
5. **P5** Lombardy。  
6. **P6** 表图打包。

---

## 4. 给各座位的一句话

| 座位 | 做什么 |
| --- | --- |
| urbansem Sync | 只实现当前票；路径+CLI 交给 Server；不拉大数据到 F: |
| Server Sync | myserver pull/跑/贴产物与闸门字段；数据不出服务器 |
| Codex Sync | 对照 draft2+本规划验收；切下一票；不写业务实现 |
| clone bot | 本地 code-only / 桥接环境，不碰数据集 |

---

*本版由 Codex Sync 根据 draft2 + EXPERIMENT_PLAN 整理（本机 Codex 会话仍在细化中时可替换为会话终稿）。*
