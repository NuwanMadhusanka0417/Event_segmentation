# HD-EMS

Hyperdimensional Event Motion Segmentation — VSA-based event camera flow and segmentation.

## Quick start

```bash
cd Train_based
pip install -r requirements.txt
pytest tests/ -v
```

## Build order (from SPEC)

| Step | Component | Gate |
|------|-----------|------|
| 1 | `vsa/fpe.py` | roundtrip > 0.99 |
| 2 | `vsa/kernel.py` | 99% energy at r≈86 |
| 3 | `vsa/field.py` | separable == explicit < 1e-10 |
| 4 | `test_retrieval.py` | **≤1 px at M=7 — GO/NO-GO** |
| 5+ | Data, model, train | see `docs/SPEC.md` |

## Project layout

```
Train_based/
├── docs/SPEC.md          Full build specification
├── configs/              YAML configs
├── hdems/                Python package
│   ├── vsa/              FPE, kernel, field, temporal (CRITICAL)
│   ├── data/             Time surfaces, DSEC, EV-IMO loaders
│   ├── models/           Encoder, matcher, decoder, segmentation
│   ├── losses/
│   ├── train.py
│   ├── eval.py
│   └── benchmark.py
├── tests/                Gate tests (run before model work)
└── scripts/              Data preparation
```

## Training

```bash
# Flow (DSEC)
python -m hdems.train --config configs/dsec_flow.yaml

# Segmentation (EV-IMO)
python -m hdems.train --config configs/evimo_seg.yaml
```

## Key design constraints

- FPE via `exp(1j * x * phases)` — never `ifft(fft(X)**x)`
- Bundled field M ≤ 9; extend range via pyramid, not larger M
- Separable 2-pass field is exact — do not replace with M² loop

See `.cursorrules` and `docs/SPEC.md` for full details.
