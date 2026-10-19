# GA-DCHG: Geometry-Anchored Dynamic Cross-Domain Hypergraphs

Code and manuscript for "Geometry-Anchored Dynamic Cross-Domain Hypergraphs for Bearing Fault Diagnosis under Load Shift."

## Structure

- `src/` — Python implementation (PyTorch, CPU-friendly)
- `results/` — JSON experiment outputs
- `manuscript_gad/` — LaTeX source for the paper
- `figures/` — figure assets

## Quick start

```bash
cd src
python exp_cross.py caches      # build hypergraph caches
python exp_cross.py screen      # run anchor screening
python exp_cross.py buildgad    # build anchor-guided caches
python exp_cross.py train       # train all models x seeds
python exp_cross.py audit       # run intervention audit
```

## Data

Requires the CWRU bearing dataset (http://www.eecs.case.edu/laboratory/bearing) and the MFPT dataset (https://www.mfpt.org/fault-data-sets/). Set paths in `src/config.py`.

## Citation

```
Zhao, Z., Li, J., and Cui, J., Geometry-Anchored Dynamic Cross-Domain Hypergraphs for Bearing Fault Diagnosis under Load Shift. Current Science, 2026 (submitted).
```
