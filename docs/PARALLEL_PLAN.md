## 1. Dependency Graph

```text
P1 metric MITMA distances (DONE)
        │
        ├── MITMA metric-distance gate
        │
        ├── MITMA canonical HBW R
        │        └── MITMA T partitions
        │
        ├── P2 Condition S
        │        └── day stacks, λR, λRᵀ, Δ, null, bootstrap
        │
        ├── P3 Condition G zero-shot
        │        └── gravity, radiation, classical controls, Deep Gravity
        │
        └── P4 neuroGravity few-shot
                 └── HBW pretraining → declared target-edge masks → held-out scoring

Lombardy data + Lombardy R/T partitions + Lombardy metric distances
        └── P5 Lombardy Condition S + Condition G

P2/P3/P4/P5 canonical outputs
        └── P6 evaluation, tables, figures, QA package
```

The shared MITMA prerequisites are:

- Materialised HBW reference matrix `R`.
- Frozen target matrices and partition definitions `T`.
- Metric distance matrix `D`, including the specified intra-zonal distances.
- Stable zone, population, and OD-key joins.
- Immutable configuration and input manifest.

P2 needs the day-level MITMA stack for daily scoring and bootstrap. P3 needs `R`, target outflows `O_i^T`, target partitions, and `D`; it does not need P2’s scores. P4 needs the same frozen inputs plus the neuroGravity adapter and declared masks. P5 is scientifically independent of the MITMA branch once Lombardy data, joins, OSM features, and metric distances are ready.

## 2. Parallel Waves

### Wave A: Read-only preparation and gate checks

These jobs can run concurrently:

- MITMA readiness and join audit.
- MITMA distance and intra-zonal-distance validation.
- Lombardy data and zone/population/OD join audit.
- Lombardy metric CRS and distance validation.
- OSM feature extraction and feature-order audit for both datasets.
- Deep Gravity and neuroGravity adapter/provenance checks.
- P6 table/figure schema validation using synthetic or contract-level inputs only.

These jobs must not modify shared canonical inputs. They should emit independent manifests and QA records.

### Wave B: Main scientific runs after immutable input freeze

Once each dataset’s `R`, `T` partitions, `D`, and manifests are frozen, these jobs can run concurrently:

- **P2 MITMA Condition S**: all specified partitions, null construction, and day bootstrap.
- **P3 MITMA Condition G zero-shot**: classical generators, Deep Gravity, and production-constrained evaluation.
- **P5 Lombardy Condition S**.
- **P5 Lombardy Condition G**, provided Lombardy OSM and model inputs are ready.
- Spatial-fold preparation and fold-integrity checks for the mechanism subset.
- Few-shot mask generation and leakage audits.

P3 and P2 do not consume each other’s statistical outputs. P4’s mask preparation can run beside both.

### Wave C: NeuroGravity and derived analyses

After the shared inputs are frozen and the neuroGravity backend contract passes:

- **P4 HBW neuroGravity pretraining** can run in parallel with P3’s independent zero-shot models.
- Within P4 itself, the sequence remains serial:
  1. HBW pretraining.
  2. Apply each declared few-shot mask.
  3. Score only held-out target cells.
- MITMA and Lombardy neuroGravity jobs can run concurrently if their feature manifests and checkpoints are independent.
- Scale analyses can run in parallel across datasets after native Condition S outputs are frozen.
- Attractiveness regressions can run in parallel with model evaluation on the mechanism subset.

P6 remains downstream of all required dataset/model jobs.

## 3. Runners Urbansem Should Implement Next

The next runners should be separate at the scientific-contract level:

1. **MITMA Condition S runner**  
   Reads frozen `R`, target partitions, day stack, null configuration, and bootstrap configuration. Produces partition-level overlap metrics, null distributions, bootstrap intervals, and QA.

2. **MITMA Condition G zero-shot runner**  
   Fits only on HBW `R`; uses target data only for legal origin outflows `O_i^T` and final scoring. Runs gravity variants, radiation, classical controls, Deep Gravity, and the declared IPF/oracle diagnostics.

3. **NeuroGravity pretraining runner**  
   Trains the HBW model on the declared spatial training split and records checkpoint, seed, feature order, and provenance.

4. **NeuroGravity few-shot runner**  
   Consumes the frozen HBW checkpoint and draft2 masks. It must enforce held-out target cells and spatial-block test origins as unavailable during adaptation.

5. **Lombardy Condition S runner**  
   Uses the same scoring contract as MITMA, while explicitly omitting day bootstrap because Lombardy has no calendar stack.

6. **Lombardy Condition G runner**  
   Uses the same production-constrained generation contract, with Lombardy-specific motive, time-band, mode, population, and OSM mappings.

7. **Unified validation/export runner**  
   Reads only completed canonical outputs and generates the draft2 table and figure source datasets. It should reject incomplete, blocked, diagnostic-only, or provenance-incomplete runs.

Each runner should have immutable inputs, an independent run record, deterministic seeds, explicit failure status, and no shared mutable intermediate files.

## 4. Serial Boundaries

These steps must remain serial:

- MITMA day-stack materialisation must finish and be atomically frozen before P2 scoring or any consumer reads it.
- `R` and `T` partition definitions must be frozen before P2, P3, or P4 scoring.
- Metric-distance validation must pass before any mainline spatial interpretation.
- NeuroGravity few-shot adaptation must follow HBW pretraining.
- Held-out scoring must follow mask construction and leakage audit.
- P6 table/figure export must follow completion and validation of all relevant upstream runs.
- Any rerun after changing data, configuration, model version, or distance computation requires a new input manifest and a new run identity.

## 5. Risks of Incorrect Parallelisation

- Reading P2’s partially materialised day stack can create missing-day, duplicate-day, or mixed-version results.
- Running P3 or P4 against a mutable `R` or partition manifest can make model comparisons non-reproducible.
- Allowing target cells into G training or few-shot adaptation causes scientific leakage.
- Reusing spatial-block test origins as few-shot observations invalidates the held-out evaluation.
- Running Lombardy with MITMA’s day-bootstrap assumptions creates unsupported uncertainty estimates.
- Exporting P6 tables while some runs are blocked can silently mix valid, provisional, diagnostic, and failed evidence.
- Sharing mutable checkpoints or feature caches across jobs can make seeds and provenance ambiguous.
- Treating a provisional distance matrix as final can invalidate all spatial conclusions even if the jobs complete successfully.

## Recommended Immediate Fan-Out

While P2 is materialising MITMA day stacks:

1. Keep P2 as the sole writer of the day-stack materialisation and publish an immutable completion marker only after validation.
2. Run MITMA readiness, join, distance, and partition-contract audits in parallel, without consuming incomplete P2 outputs.
3. Prepare and validate the P3 zero-shot runner against the frozen HBW schema, then launch only after `R`, target partitions, and `D` are frozen.
4. Prepare neuroGravity checkpoints, masks, and leakage audits in parallel; defer target adaptation until HBW pretraining inputs are frozen.
5. Start the independent Lombardy readiness, metric-distance, OSM, Condition S, and classical Condition G branch as soon as its own inputs pass.
6. Prepare P6 export contracts in parallel, but block all paper tables and figures until P2–P5 outputs pass their respective gates.

This gives the highest safe fan-out while preserving the required scientific serial points.