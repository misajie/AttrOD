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

## License

MIT
