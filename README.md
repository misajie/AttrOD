# AttrOD

Experiment codebase for **Condition S** (observed-overlap of attribute-specific OD cuts vs morning HBW) and **Condition G** (generators fitted only on HBW, emitted production-constrained onto target outflows). Spec source: `draft2.md`.

## Install

```bash
cd AttrOD
python -m venv .venv && source .venv/bin/activate   # or your env
pip install -e ".[dev]"
# optional: pip install -e ".[torch,geo]"
```

Import check (no data required):

```bash
python -c "from attrOD.metrics import cpc, cpl, cpcd, pearson_log, nrmse, jsd, delta_cpc; print('ok', cpc.__name__)"
python -c "from attrOD.condition_s import score_condition_s; from attrOD.condition_g import score_condition_g; print('pipelines ok')"
```

## Expected data directories (not shipped)

Place product extracts under `data/` (or override paths in study-area YAML):

```
data/
  mitma/
    districts/          # district polygons + id crosswalk
    daily_trips/        # vendor daily trip files (date, hour, od, activities, …)
    population/         # INE → district population / jobs
  lombardy/
    zones/              # OD2016 zone polygons
    matrice_od2016/     # motive × mode × time-band tables
    population/         # ISTAT / Regione attributes
  osm/
    extracts/           # dated .osm.pbf clips (5 km buffer)
  study_areas/          # optional zone id lists per area
```

Configs live in `configs/study_areas/*.yaml` (stubs; fill Table 1 / S1 fields — do not invent city lists). S1 metadata schema: `configs/s1_schema.yaml`.

## CLI (Server Bot recipe)

After install and data placement:

```bash
# 1) Build long-table partitions + reference R
python scripts/build_partitions.py --config configs/study_areas/<area>.yaml --out outputs/<area>/partitions

# 2) Queen contiguous 5-block origin split (mechanism subset, N≥80)
python scripts/make_spatial_blocks.py --config configs/study_areas/<area>.yaml --out outputs/<area>/blocks.json

# 3) Condition S (λR, λRᵀ, independence null; day bootstrap if D≥2)
python scripts/run_condition_s.py --config configs/study_areas/<area>.yaml --partitions outputs/<area>/partitions --out outputs/<area>/condition_s

# 4) Condition G (fit on R train origins; emit T̂_ij = O_i^T p_j|i; IPF variants)
python scripts/run_condition_g.py --config configs/study_areas/<area>.yaml --partitions outputs/<area>/partitions --blocks outputs/<area>/blocks.json --out outputs/<area>/condition_g

# 5) Coarser tessellations for Table 8
python scripts/aggregate_tessellation.py --config configs/study_areas/<area>.yaml --mode official_dissolve --out outputs/<area>/scale
```

No datasets are downloaded by this package. Synthetic smoke tests under `tests/` cover metrics and the production-constrained row-sum assertion only.


## S1 — Hong Kong Teralytics materialisation

Pipe-delimited vendor matrix + CHKU shapefile DBF centroids (WGS84). Strict 07–10 HBW is **unavailable**; reference proxy is weekdays ∩ `PartOfDay=4:00-12:00` ∩ `TripPurpose=to-work`.

```bash
# Place extracts (not shipped):
#   data/hk/OD_Matrix_Teralytics/teralytics_matrix.csv
#   data/hk/shapefile/CHKU_Shapes.{dbf,shp,prj}

python scripts/process_hk_teralytics.py \
  --csv data/hk/OD_Matrix_Teralytics/teralytics_matrix.csv \
  --shapefile-dir data/hk/shapefile \
  --out outputs/hk_teralytics \
  --chunksize 500000
```

Artefacts under `--out`: `zones.csv`, `long_total.csv.gz` (or `.parquet`), `total.npy`, `R_HBW_proxy.npy`, `distance_provisional.npy` (label `provisional_wgs84_not_metric`), `qa.json`, `manifest.json`, optional `partitions/`. TOTAL aggregation excludes vendor `All` roll-ups on Age/Gender/TripDistance/TripDuration when present (see `qa.json` → `total_aggregation`). Shapefile FID set is 1–8,10–32 (FID 9 missing); unmatched matrix ids are listed in `qa.json`.

Config stub: `configs/study_areas/hk_teralytics.yaml`.

## Package map → draft2

| Module | draft2 section |
|--------|----------------|
| `attrOD.data` | Data / products / Flow long table |
| `attrOD.tessellation` | Distances, inclusion, dissolve/merge |
| `attrOD.partitions` | Purpose×time, day/sex/age, length bands, … |
| `attrOD.metrics` | CPC, CPL, CPCd, Pearson/R², NRMSE, JSD, Δ |
| `attrOD.models` | Gravity, radiation, Deep Gravity, RF, closed-form hook, meta/neuroGravity, IPF, oracle |
| `attrOD.condition_s` | Condition S pipeline |
| `attrOD.condition_g` | Condition G pipeline |
| `attrOD.spatial` | 5-block queen contiguous splits |
| `attrOD.scale` | Official dissolve + random contiguous merge |
| `attrOD.reporting` | Table schemas 1–8 |


## Tickets 1–3 (server readiness, Condition S engine, Condition G registry)

Hard rules: no invented cities/Table 1; no Mac `data_root`; HK always `verification_only=true` / `paper_mainline=false`; DeepGravity pin `8693536`, NeuroGravity pin `7e29ef0`; NeuroGravity stub ≠ formal main results.

```bash
# 0) Data readiness (myserver: --data-root ~/AttrOD/data)
python scripts/audit_data_readiness.py \
  --data-root ~/AttrOD/data \
  --out outputs/readiness \
  --hk-qa outputs/hk_teralytics/qa.json

# On this box only, backup root is allowed explicitly:
python scripts/audit_data_readiness.py \
  --data-root /workspace/AttrOD/data --allow-backup \
  --out outputs/readiness --hk-qa outputs/hk_teralytics/qa.json

# 1) HK vertical smoke (Condition S + Gravity_power + O_i^T QA)
python scripts/run_hk_smoke.py \
  --data-root ~/AttrOD/data \
  --hk-materialised outputs/hk_teralytics \
  --out outputs/hk_smoke \
  --bootstrap-resamples 1000

# 2) Condition S exact engine (matrices)
python scripts/run_condition_s.py --config configs/study_areas/hk_teralytics.yaml \
  --partitions outputs/hk_teralytics --T outputs/hk_teralytics/total.npy \
  --R outputs/hk_teralytics/R_HBW_proxy.npy --out outputs/hk_smoke/condition_s \
  --verification-only --allow-backup --data-root /workspace/AttrOD/data

# 3) Condition G registry + adapter dry-run
python scripts/run_condition_g.py --config configs/study_areas/hk_teralytics.yaml \
  --partitions outputs/hk_teralytics --out outputs/hk_smoke/condition_g \
  --R outputs/hk_teralytics/R_HBW_proxy.npy --T outputs/hk_teralytics/total.npy \
  --distances outputs/hk_teralytics/distance_provisional.npy \
  --attractiveness outputs/hk_teralytics/R_HBW_proxy.npy --registry \
  --models Gravity_power,Radiation,IPF_G-row,Oracle --verification-only

python scripts/run_condition_g.py --config configs/study_areas/hk_teralytics.yaml \
  --partitions outputs/hk_teralytics --out outputs/hk_smoke/condition_g \
  --adapters-dry-run
```

Stub vs ready:
- **Ready:** manifest/readiness validators, Condition S λ/Δ/null/day-bootstrap, partitions audit, O_i^T project/audit, classic registry (gravity/radiation/RF/IPF/oracle/closed-form/meta-Gravity), HK smoke CLI, unit tests with synthetic fixtures.
- **Stub / reserved:** `NeuroGravityAdapter(backend="stub")` (default); `backend="official"` reserved until torch_geometric / integration decision — never paper-mainline. DeepGravity uses package hyperparameters with read-only third_party pin audit; full official training loop wiring is adapter-mediated.

## License

MIT
