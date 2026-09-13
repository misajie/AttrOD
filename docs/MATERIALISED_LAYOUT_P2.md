# MITMA Condition S materialised layout (P2)

Place these arrays on the compute host (paths illustrative):

- `R.npy` — morning HBW reference (weekdays, AM home→work_or_study), N×N
- `distances.npy` — accepted metric D (EPSG:25830), N×N
- `distance_qa.json` — must include `"distance_status": "metric_projected"`
- `T_full.npy` or `total.npy` — unsliced weekday total (or analysis-window total)
- `T_other__AM.npy`, `T_home__PM.npy`, `T_home__AM.npy` — draft2 purpose–time cuts
- optional length cuts: `T_length__0.5-2.npy`, …
- optional day stacks: `R_by_day/*.npy`, `T_by_day/*.npy` (≥2 weekdays for bootstrap)
- optional `zone_ids.json`

Run:

```bash
attrOD-run-mitma-condition-s \
  --materialised <dir> \
  --partitions-yaml configs/condition_s/draft2_partitions.yaml \
  --out outputs/mainline/mitma/condition_s \
  --bootstrap-resamples 1000 --seed 0
```
