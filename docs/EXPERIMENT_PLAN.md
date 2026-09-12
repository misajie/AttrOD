# AttrOD 实验 CODE PLAN

Source: Codex `gpt-6-astra` / ultra (`reasoning.effort=max`)
Session: `01a09415-8ff7-7c60-8083-4349a1a4fc32`
Role: plan only — urbansem writes; UrbanSem Server Bot runs on myserver.

---

以下计划严格以 brief 为边界。`draft2` 的具体表号和图号应由 `urbansem` 从 `draft2.md` 回填；本计划只使用语义锚点，不新增城市、Table 1 条目或论文数据。所有数据实验均在 `myserver` 执行，`~/AttrOD/data/` 是唯一 canonical root；`/workspace/AttrOD/data` 只作为备份源，数据不得落到用户 Mac。

## 1. Goal & non-goals

目标是先完成一个可审计、可复现的最小纵向实验切片，再扩展到完整 MITMA 和 Lombardy 主线：Condition S 计算 `λR`、`λRᵀ`、`Δ`、independence null 与 day bootstrap；Condition G 严格只在 HBW `R` 上训练生成器，并将输出约束到 production-constrained `O_i^T`。HK 仅用于今日 smoke、接口和数据契约验证，不进入论文主线；在距离仍为 `provisional_wgs84_not_metric`、FID 9 缺失或 daily trips 不完整时，不产生主线结论。

## 2. Work packages

### NOW-0：服务器数据与运行契约

- Owner：`urbansem` 写数据契约和 validator；`UrbanSem Server Bot` 在 `myserver` 执行；`Codex Bot` 只计划并接受下载结果、校验清单和 checksum。
- Inputs：
  - `~/AttrOD/data/hk`
  - `~/AttrOD/outputs/hk_teralytics`
  - `~/AttrOD/data/mitma/districts`
  - `~/AttrOD/data/mitma/population`
  - `~/AttrOD/third_party`
- Outputs：
  - `~/AttrOD/outputs/readiness/server_manifest.json`
  - `~/AttrOD/outputs/readiness/dataset_readiness.json`
  - 每次 run 的配置、代码版本、数据 checksum、环境和数据 root 记录
- Acceptance：
  - `data_root` 只能解析为 `~/AttrOD/data`。
  - `/workspace/AttrOD/data` 若被使用，必须在 manifest 中标记为 backup，并禁止与 canonical root 混用。
  - 每个输入文件有路径、大小、hash、时间范围、schema 和来源记录。
  - 明确记录 `daily_trips` 的完整性、Lombardy/OSM 是否为空、HK FID 9 缺失。
  - 发现 Mac 路径时直接失败。
- Compute：CPU 2–4 cores，8–16 GB RAM，无 GPU。

### NOW-1：HK Condition S smoke

- Owner：`urbansem` 写 Condition S 计算和分区接口；`Server Bot` 运行。
- Inputs：
  - `~/AttrOD/data/hk`
  - `~/AttrOD/outputs/hk_teralytics/qa.json`
  - `~/AttrOD/outputs/hk_teralytics/R_HBW_proxy`
  - 当前距离状态 `provisional_wgs84_not_metric`
  - FID 映射，其中 FID 9 缺失
- Outputs：
  - `~/AttrOD/outputs/hk_smoke/condition_s/metrics.json`
  - `~/AttrOD/outputs/hk_smoke/condition_s/partition_metrics.parquet`
  - `~/AttrOD/outputs/hk_smoke/condition_s/day_bootstrap.parquet`
  - `~/AttrOD/outputs/hk_smoke/condition_s/qa.json`
  - `~/AttrOD/outputs/hk_smoke/run_manifest.json`
- Acceptance：
  - 正确生成 `R`、`Rᵀ`、`λR`、`λRᵀ`、`Δ` 及 draft2 要求的 independence null 字段。
  - day bootstrap 以 day 为重采样单位，保留日内 OD 结构。
  - FID 9 必须被显式 quarantine 或排除并记录原因，不得静默填补。
  - provisional distance 只允许做 schema、分区、数值稳定性和接口验证。
  - 输出明确写入 `verification_only=true`、`paper_mainline=false`。
  - 不允许从 HK smoke 输出论文效果结论。
- Compute：CPU 4–8 cores，16–32 GB RAM，无 GPU。

### NOW-2：HK Condition G 最小生成器切片

- Owner：`urbansem` 写生成器统一接口和约束检查；`Server Bot` 运行。
- Inputs：
  - HK 的 `R_HBW_proxy`
  - NOW-1 生成的分区和运行 manifest
  - package 中已有 gravity、radiation、RF、IPF、oracle、closed-form hook
- Outputs：
  - `~/AttrOD/outputs/hk_smoke/condition_g/{model}/predictions.parquet`
  - `~/AttrOD/outputs/hk_smoke/condition_g/{model}/metrics.json`
  - `~/AttrOD/outputs/hk_smoke/condition_g/{model}/constraint_qa.json`
- Acceptance：
  - 所有模型的训练输入只来自 HBW `R`。
  - 生成结果必须经过 production constraint，最终落在 `O_i^T` 的合法域内。
  - 检查生产约束、非负性、键完整性、重复 OD、NaN/Inf 和输出维度。
  - oracle 只能标记为 upper-bound/diagnostic，不能当作可部署模型。
  - HK 结果全部标记 verification-only。
- Compute：经典模型 CPU 4–8 cores，16–32 GB RAM；若运行 DeepGravity 或 NeuroGravity dry-run，最多 1 张 16–24 GB GPU。

### NOW-3：DeepGravity / NeuroGravity adapter 审计

- Owner：`urbansem` 写 adapter、版本和输入输出契约；`Server Bot` 做 import、依赖和最小推理检查。
- Inputs：
  - `~/AttrOD/third_party` 中 DeepGravity commit `8693536`
  - `~/AttrOD/third_party` 中 NeuroGravity commit `7e29ef0`
  - HK smoke schema
- Outputs：
  - `~/AttrOD/outputs/third_party/deepgravity_manifest.json`
  - `~/AttrOD/outputs/third_party/neurogravity_manifest.json`
  - `~/AttrOD/outputs/third_party/adapter_qa.json`
- Acceptance：
  - 记录真实 commit、依赖版本、特征顺序、归一化规则、随机种子和 checkpoint。
  - adapter 能把 AttrOD 的 HBW `R` 输入转换为第三方模型格式，并把输出转换回 AttrOD canonical OD schema。
  - 任何模型输出均经过同一套 `O_i^T` 约束检查。
  - NeuroGravity 轻量 stub 只能用于 smoke/接口诊断。
  - full edge-enhanced GT 在 `torch_geometric` 和官方 NeuroGravity 集成方案未决定前，不能进入论文主结果。
- Compute：CPU 2–4 cores，8–16 GB RAM；有训练时使用 1 张 16–24 GB GPU。

### NOW-4：运行编排与可复现性

- Owner：`urbansem` 写 CLI、配置加载和 run ledger；`Server Bot` 运行。
- Inputs：NOW-0 至 NOW-3 的配置、代码版本和数据 manifest。
- Outputs：
  - `~/AttrOD/outputs/runs/<run_id>/run_manifest.json`
  - `~/AttrOD/outputs/runs/<run_id>/status.json`
  - `~/AttrOD/outputs/runs/<run_id>/logs/`
- Acceptance：
  - 每个 run 可由一个唯一 `run_id` 重放。
  - 运行失败时保留失败阶段、输入 hash 和完整错误信息。
  - 并行任务不会共享可变中间文件。
  - 输出文件名、schema 和随机种子固定。
- Compute：CPU 2–4 cores，8–16 GB RAM，无 GPU。

### WHEN-1：MITMA daily trips 完整性通过后的主线

- Owner：`urbansem` 写 MITMA loader 和主线配置；`Server Bot` 在 `myserver` 运行。
- Inputs：
  - `~/AttrOD/data/mitma/daily_trips`
  - `~/AttrOD/data/mitma/districts`
  - `~/AttrOD/data/mitma/population`
  - 通过距离和 FID gate 的空间输入
- Outputs：
  - `~/AttrOD/outputs/mainline/mitma/condition_s/`
  - `~/AttrOD/outputs/mainline/mitma/condition_g/`
  - `~/AttrOD/outputs/mainline/mitma/figures/`
  - `~/AttrOD/outputs/mainline/mitma/tables/`
- Acceptance：
  - 2022-10 的 31 天全部存在、可解析、无重复日、无 partial day。
  - 每天的 zone key、人口 join、OD key 和单位一致。
  - 所有距离为 metric CRS；FID 全量闭合。
  - Condition S 的 partition、null 和 bootstrap 与 HK smoke 使用同一契约。
  - Condition G 只用 HBW `R` 训练，`O_i^T` 约束在推理后强制检查。
- Compute：CPU 16–32 cores，64–128 GB RAM；经典模型可无 GPU；DeepGravity/NeuroGravity 训练建议 1–2 张 24 GB GPU。

### WHEN-2：Lombardy 与 OSM 数据到位后的主线

- Owner：`urbansem` 写 Lombardy/OSM schema adapter；`Server Bot` 通过已批准 proxy 获取并运行；`Codex Bot` 只接受下载 manifest 和 checksum。
- Inputs：
  - `~/AttrOD/data/lombardy`
  - `~/AttrOD/data/osm`
  - package 中的 Lombardy matrices、zones、population 对应输入
  - OSM Spain 和 italy-nord-ovest PBF
- Outputs：
  - `~/AttrOD/outputs/mainline/lombardy/condition_s/`
  - `~/AttrOD/outputs/mainline/lombardy/condition_g/`
  - `~/AttrOD/outputs/mainline/lombardy/figures/`
  - `~/AttrOD/outputs/mainline/lombardy/tables/`
- Acceptance：
  - download manifest、checksum、来源和时间戳完整。
  - zones、population、OD、OSM feature 的 join 全闭合。
  - 距离计算使用明确的 metric CRS；禁止沿用 HK provisional WGS84 距离。
  - OSM 解析后的特征顺序和单位在 run manifest 中固定。
  - Lombardy 结果与 MITMA 结果使用相同的 Condition S/G 评估协议。
- Compute：CPU 16–32 cores，64–128 GB RAM；OSM 解析可临时需要 128 GB RAM；深度模型建议 1–2 张 24 GB GPU。

### WHEN-3：全量模型矩阵与 few-shot NeuroGravity

- Owner：`urbansem` 写模型 registry、few-shot mask 和统一评估；`Server Bot` 批量运行。
- Inputs：
  - 通过 WHEN-1/2 gate 的 MITMA 和 Lombardy 数据
  - draft2 规定的 few-shot mask 配置
  - DeepGravity、NeuroGravity 固定 commit
- Outputs：
  - `~/AttrOD/outputs/mainline/{dataset}/condition_g/{model}/{mask}/`
  - 每个模型/掩码的预测、指标、约束 QA、训练日志和 checkpoint manifest
- Acceptance：
  - mask 只作用于 HBW `R` 的训练输入，禁止使用目标侧信息泄漏。
  - few-shot 的 k、mask 形状和划分严格来自 draft2 配置，不自行增加实验档位。
  - 主表中的模型均有可重放的 config、seed、checkpoint 和输入 hash。
  - NeuroGravity stub 与正式 edge-enhanced 版本在结果和表格中明确分列，不混淆。
- Compute：每个深度模型 1–2 张 24 GB GPU；经典基线 CPU 16 cores 即可。

## 3. Experiment matrix mapped to draft2 Tables/Figs

实际表号、图号由 `draft2.md` 的最终编号回填；以下是实验语义和输出锚点。

| draft2 语义锚点 | 数据/划分 | 实验内容 | 必备输出 |
|---|---|---|---|
| Condition S 主表 | MITMA、Lombardy；HK 仅 smoke | 原始 `R` 与转置 `Rᵀ` 的 `λR`、`λRᵀ`、`Δ` | 点估计、draft2 要求的区间/显著性字段、数据与分区 manifest |
| Condition S partition 对比 | draft2 规定的 full、spatial block、scale、day 划分 | 每个合法 partition 独立计算统计量 | partition-level parquet、分区覆盖率、重叠审计 |
| Condition S null 图/表 | 同一 partition | independence null 的观测统计量、null 分布及比较 | null samples、阈值/区间、校准 QA |
| Condition S bootstrap 图/表 | 以 day 为单位 | day bootstrap，保留日内 OD 依赖 | bootstrap 分布、置信区间、seed 和重采样清单 |
| Condition G 生成器主表 | MITMA、Lombardy；只从 HBW `R` 训练 | gravity、radiation、RF、IPF、oracle、closed-form hook、DeepGravity、NeuroGravity | 统一指标、约束检查、训练/推理时间、模型 manifest |
| Condition G 生成器图 | 同一数据与 partition | 观测 HBW 输入、生成输出和 production-constrained `O_i^T` 的对比 | 可复现的预测文件、图形源数据、约束 QA |
| Few-shot NeuroGravity 图/表 | draft2 指定 mask | 不同 few-shot mask 下的 NeuroGravity 对比 | mask manifest、无泄漏审计、stub/正式模型分列 |
| 附录 QA | 所有主线数据 | 距离、FID、population join、daily completeness、数据 checksum | `readiness.json`、`constraint_qa.json`、失败项清单 |

必须保持以下矩阵约束：

1. HK 不出现在论文主线表格和结论中。
2. Condition S 的所有模型无关统计量必须使用同一 partition 定义、同一 day bootstrap 规则和同一 null 构造。
3. Condition G 的训练输入只能是 HBW `R`；production constraint 作用于输出阶段并写入 QA。
4. oracle 只作为上界/诊断。
5. few-shot NeuroGravity 的 mask、k 和划分来自 draft2，不能由实现者临时设计。
6. 不得用 provisional WGS84 距离生成主线空间结论。

## 4. Code gaps to implement

以下只定义模块、函数名和接口契约，不包含实现代码。

### 数据与运行契约

- `data.manifest.build_run_manifest`
  - 输入：dataset root、配置、代码 commit、第三方 commit、环境信息。
  - 输出：不可变 run manifest，包含路径、hash、schema、时间范围和 `paper_mainline` 标志。
- `data.manifest.validate_server_data_root`
  - 输入：数据 root。
  - 输出：通过/失败及违规路径；拒绝 Mac 路径和未声明 backup root。
- `data.readiness.validate_daily_trips_completeness`
  - 输入：daily trips root、预期日期集合。
  - 输出：完整日期、缺失日期、partial day、重复日和 checksum。
- `data.readiness.validate_zone_population_join`
  - 输入：zones、population、OD keys。
  - 输出：join coverage、孤立 key、重复 key 和单位一致性结果。
- `spatial_blocks.distance_gate.validate_metric_distance`
  - 输入：坐标、CRS、距离文件。
  - 输出：metric CRS 证明、距离单位、异常值统计；provisional 状态直接阻止 mainline。

### Condition S

- `partitions.condition_s.build_partitions`
  - 输入：dataset manifest、draft2 partition config。
  - 输出：带版本号的 partition index。
- `partitions.condition_s.audit_partition_disjointness`
  - 输入：partition index。
  - 输出：覆盖率、重叠、空分区和边界审计。
- `metrics.condition_s.compute_lambda_r`
  - 输入：`R`、partition、draft2 参数。
  - 输出：`λR` 及中间统计量。
- `metrics.condition_s.compute_lambda_rt`
  - 输入：`Rᵀ`、同一 partition、同一参数。
  - 输出：`λRᵀ`。
- `metrics.condition_s.compute_delta`
  - 输入：`λR`、`λRᵀ`。
  - 输出：`Δ`，保留定义版本和数值精度。
- `metrics.condition_s.evaluate_independence_null`
  - 输入：观测统计量、draft2 null 构造、随机种子。
  - 输出：null 分布、比较结果和校准 QA。
- `metrics.condition_s.day_bootstrap`
  - 输入：按 day 分组的 OD 记录、bootstrap 配置、seed。
  - 输出：重采样索引、统计量分布和区间。

### Condition G

- `condition_g.masks.build_few_shot_masks`
  - 输入：HBW `R`、draft2 mask 配置、seed。
  - 输出：mask manifest；必须能证明 mask 未使用目标侧信息。
- `condition_g.generators.fit_hbw_generator`
  - 输入：HBW `R`、特征、partition、模型 config。
  - 输出：模型 artifact、训练 manifest、可重放状态。
- `condition_g.generators.generate_production_constrained`
  - 输入：训练好的模型、输入特征、production constraints。
  - 输出：落在 `O_i^T` 的预测 OD 和 constraint report。
- `condition_g.constraints.project_to_O_i_T`
  - 输入：未约束预测、生产约束、目标 schema。
  - 输出：合法预测、投影前后差异和失败原因。
- `condition_g.constraints.audit_constraints`
  - 输入：预测 OD、`O_i^T`、production constraints。
  - 输出：非负性、总量、row/column 约束、key 完整性和容差检查。
- `evaluation.condition_g.evaluate_generators`
  - 输入：预测、观测、draft2 指标配置、partition。
  - 输出：模型级指标、分区指标、运行时和失败记录。

### 模型 registry 与第三方

- `models.registry.register_model`
  - 输入：模型名、版本、训练/推理 adapter、所需资源。
  - 输出：可查询的模型 registry entry。
- `models.adapters.deep_gravity.DeepGravityAdapter`
  - 契约：AttrOD HBW schema ↔ pinned DeepGravity schema；输出必须回到 canonical OD schema。
- `models.adapters.neuro_gravity.NeuroGravityAdapter`
  - 契约：区分 lightweight stub 和正式 edge-enhanced backend；两者不能共用论文主结果标签。
- `models.adapters.normalize_features`
  - 契约：固定特征顺序、单位、缺失值规则和训练/推理一致性。
- `models.adapters.record_third_party_provenance`
  - 契约：写入 commit、依赖、checkpoint、seed 和输入 hash。

### CLI 与产物

- `scripts.audit_data_readiness`
- `scripts.run_condition_s`
- `scripts.run_condition_g`
- `scripts.run_hk_smoke`
- `scripts.run_mainline_dataset`
- `scripts.export_draft2_tables`
- `scripts.export_draft2_figures`
- `scripts.validate_run_outputs`

每个 CLI 都必须支持：配置路径、run id、数据 root、输出 root、seed、dry-run、失败即停和 manifest 输出。

## 5. Integration with third_party DeepGravity / NeuroGravity

采用 wrapper/adaptor 方案，不重写第三方核心算法。

DeepGravity：

- 固定使用 commit `8693536`。
- AttrOD 负责数据 schema、HBW-only split、特征顺序、归一化、seed、checkpoint manifest 和 `O_i^T` 输出约束。
- 第三方代码保持只读；只在 adapter 层处理路径、参数和输出格式。
- 结果必须同时保存第三方原始输出和 AttrOD canonical 输出，便于审计。

NeuroGravity：

- 固定使用 commit `7e29ef0`。
- 当前 package 中的实现是 meta-Gravity 加轻量 residual stub，只能用于 HK smoke、接口测试和依赖审计。
- full edge-enhanced GT 需要先决定采用官方 NeuroGravity 集成还是补齐 `torch_geometric` 依赖；决定前不得生成论文主结果。
- few-shot mask 必须由 AttrOD 统一生成并记录，不能由第三方 adapter 私自重写。
- 正式模型与 stub 在 registry、输出目录、表格列和最终解释中严格分开。

## 6. HK smoke path：今天可运行

1. `Server Bot` 固定 `data_root=~/AttrOD/data`，读取 `~/AttrOD/data/hk` 和现有 `~/AttrOD/outputs/hk_teralytics`。
2. 载入 `qa.json` 和 `R_HBW_proxy`，生成 HK run manifest。
3. 将 `distance_status=provisional_wgs84_not_metric` 和 `missing_fid=[9]` 写入 manifest。
4. 执行 Condition S：
   - 构造 `R` 与 `Rᵀ`；
   - 计算 `λR`、`λRᵀ`、`Δ`；
   - 执行 independence null；
   - 执行 day bootstrap；
   - 输出 partition 和数值 QA。
5. 执行 Condition G 的经典最小矩阵：
   - gravity；
   - radiation；
   - RF；
   - IPF；
   - oracle；
   - closed-form hook。
6. 对每个生成结果执行 `O_i^T` 约束检查。
7. 对 DeepGravity 和 NeuroGravity 只做 adapter import、schema round-trip 和最小 dry-run；NeuroGravity 结果标为 stub。
8. 输出到 `~/AttrOD/outputs/hk_smoke/`，不得写入 Mac 或覆盖 `hk_teralytics`。
9. 该路径的成功标准是契约、数值、分区、约束和可复现性通过；不得把 HK 结果升级为论文主线证据。

## 7. Kill / gate criteria

满足以下条件前，禁止扩展到完整 MITMA+Lombardy 主线：

- 距离仍为 `provisional_wgs84_not_metric`。
- 任意主线数据存在缺失 FID，包括 HK 的 FID 9 未明确 quarantine 时。
- MITMA daily trips 不是完整 31 天，或存在 partial day、重复日、不可解析日。
- `districts`、population、OD key 的 join coverage 未达到全闭合。
- 任何 run 混用了 `~/AttrOD/data` 与 `/workspace/AttrOD/data`，或 manifest 无法证明 canonical source。
- Condition S 的 `R`/`Rᵀ`、partition、null 或 day bootstrap 输出不稳定、不可重放或出现未解释 NaN/Inf。
- Condition G 训练输入含有非 HBW `R` 的目标侧信息。
- 生成结果无法通过 production constraint 或无法落到 `O_i^T`。
- DeepGravity commit、NeuroGravity commit、依赖或 checkpoint 未记录。
- NeuroGravity 仍只有 lightweight stub，却被标记为正式 edge-enhanced 结果。
- 运行结果缺少 config、seed、输入 hash、代码 commit 或失败日志。
- 任意表/图需要新增城市或 Table 1 条目才能成立。

触发 kill 时保留失败产物和 manifest，修复后使用新 `run_id` 重跑，不覆盖失败记录。

## 8. Suggested file/script layout

建议代码位于服务器 `~/AttrOD/` 的仓库目录中，数据和输出分离：

```text
AttrOD/
  configs/
    data/
      hk.yaml
      mitma.yaml
      lombardy.yaml
    condition_s/
      draft2_partitions.yaml
      null.yaml
      day_bootstrap.yaml
    condition_g/
      generators.yaml
      production_constraints.yaml
      few_shot_masks.yaml
    third_party/
      deepgravity.yaml
      neurogravity.yaml

  data_contracts/
    schemas/
    manifests/

  metrics/
    condition_s/
  partitions/
    condition_s/
  condition_g/
    masks/
    constraints/
    generators/
  models/
    registry/
    adapters/
  spatial_blocks/
    distance_gate/
  evaluation/
    condition_s/
    condition_g/

  scripts/
    audit_data_readiness
    run_hk_smoke
    run_condition_s
    run_condition_g
    run_mainline_dataset
    export_draft2_tables
    export_draft2_figures
    validate_run_outputs

  third_party/
    DeepGravity/
    NeuroGravity/

  outputs/
    hk_smoke/
    mainline/
      mitma/
      lombardy/
    third_party/
    readiness/
    runs/
```

`~/AttrOD/data/` 不纳入代码仓库；`/workspace/AttrOD/data` 只作为 backup 输入，不作为默认配置值。`outputs/` 中每个 run 使用独立目录和 manifest，表格/图形只从已通过 gate 的 canonical outputs 导出。

## 9. Immediate first 3 tickets for urbansem

### Ticket 1：Server-only manifest + HK vertical slice

- Owner：`urbansem` 写；`Server Bot` 跑。
- 内容：
  - 固定 canonical data root；
  - 实现 readiness manifest；
  - HK FID 9 quarantine；
  - provisional distance 明确标记；
  - 串通 Condition S smoke、一个经典 Condition G generator 和 `O_i^T` constraint QA。
- DoD：
  - `~/AttrOD/outputs/hk_smoke/` 可从零生成；
  - 无 Mac 路径；
  - 失败项可定位；
  - 输出标记 `verification_only=true`。

### Ticket 2：Condition S exact engine

- Owner：`urbansem` 写；`Server Bot` 用 HK 运行。
- 内容：
  - 实现 `λR`、`λRᵀ`、`Δ`；
  - 接入 draft2 partitions；
  - 实现 independence null；
  - 实现 day bootstrap；
  - 输出统一 parquet/json schema。
- DoD：
  - 同一输入、配置和 seed 可重放；
  - 分区覆盖和重叠审计通过；
  - null 与 bootstrap 的中间产物可审计；
  - HK 结果仍保持 verification-only。

### Ticket 3：Condition G registry + third-party adapters

- Owner：`urbansem` 写；`Server Bot` 运行经典模型和 dry-run。
- 内容：
  - 建立 gravity、radiation、RF、IPF、oracle、closed-form hook 的统一 registry；
  - 实现 HBW-only 训练契约；
  - 实现 `O_i^T` production constraint；
  - 接入 pinned DeepGravity；
  - 接入 NeuroGravity stub，并预留正式 edge-enhanced backend；
  - 接入 draft2 few-shot mask 配置。
- DoD：
  - 每个模型都有统一输入、输出、metrics 和 constraint QA；
  - no-leakage audit 通过；
  - DeepGravity commit `8693536`、NeuroGravity commit `7e29ef0` 可追溯；
  - stub 不会被误标为正式 NeuroGravity 主结果；
  - MITMA/Lombardy 主线仍被 readiness、距离和 daily completeness gates 阻断，直到对应条件满足。