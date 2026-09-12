#!/bin/bash
#SBATCH -J attrod_hk_smoke
#SBATCH -p cpu
#SBATCH -A zhangqingpeng
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 01:30:00
#SBATCH -o /public/home/jjtianhkuhpc1/AttrOD/_hk_smoke_%j.out
#SBATCH -e /public/home/jjtianhkuhpc1/AttrOD/_hk_smoke_%j.err
set -euo pipefail
cd /public/home/jjtianhkuhpc1/AttrOD
source /public/home/jjtianhkuhpc1/venv/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
pip install -e ".[dev]" --no-build-isolation --no-deps
python scripts/audit_data_readiness.py --data-root ~/AttrOD/data --out outputs/readiness --hk-qa outputs/hk_teralytics/qa.json
python scripts/run_hk_smoke.py --data-root ~/AttrOD/data --hk-materialised outputs/hk_teralytics --out outputs/hk_smoke --bootstrap-resamples 1000
python scripts/run_condition_g.py --config configs/study_areas/hk_teralytics.yaml --partitions outputs/hk_teralytics --out outputs/hk_smoke/condition_g --adapters-dry-run
echo HK_SMOKE_DONE
find outputs/hk_smoke outputs/readiness -type f | sort
