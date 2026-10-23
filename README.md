# DyNeC — Noisy Label Detection and Correction by Consensus of Training Dynamics and Latent Neighbourhoods

Code, raw results and figure scripts for the article

> G. Arul Dalton and P. Sankar. *Noisy Label Detection and Correction by Consensus of Training Dynamics and Latent Neighbourhoods.* Submitted to *Data Mining and Knowledge Discovery* (Springer).

DyNeC estimates, for every training sample, the probability that its label is correct by fusing

* a **temporal view** — the mean prediction margin along the training trajectory, and
* a **spatial view** — reliability-weighted k-nearest-neighbour label agreement in the learned embedding,

with **class-conditional two-component Gaussian mixtures** combined by a **product-of-experts log-odds rule**.
The fused posterior drives **consensus soft relabelling**, and the per-sample posteriors and corrected labels form an auditable cleaned dataset.

## Repository layout

| Path | Content |
|---|---|
| `dynec/core.py` | NumPy reference implementation: MLP, DyNeC (Algorithms 1–2), all baselines (CE, GCE, SCE, Bootstrap, APL, ENN filter, small-loss GMM), noise generators |
| `experiments/full.py` | Main comparison, ablation, sensitivity and learning-curve runs (resumable, writes `results/results.jsonl`) |
| `experiments/extra.py` | Clean-label reference runs |
| `experiments/exp2.py` | ENN baseline and transfer of corrected labels to logistic regression (`results/results2.jsonl`) |
| `analysis/analyze.py` | Aggregates raw runs into `results/summary.json` |
| `analysis/fig_blocks.py`, `analysis/fig_results.py` | Regenerate Fig. 1–6 (PNG 600 dpi and EPS) in `figures/` |
| `results/` | Raw per-run results of every number reported in the paper |
| `pytorch/dynec_torch.py` | PyTorch/GPU port for CIFAR-10/100 and CIFAR-N. **Not used for any reported result and not yet executed** |

## Quick start

```bash
git clone https://github.com/GArulDalton/DyNeC.git
cd DyNeC
pip install -r requirements.txt
bash scripts/download_data.sh          # MNIST and Fashion-MNIST; UCI Digits comes with scikit-learn
python experiments/full.py             # ~3 h on one CPU core
python experiments/extra.py
python experiments/exp2.py             # ~1 h
python analysis/analyze.py
python analysis/fig_blocks.py && python analysis/fig_results.py
```

Every run is deterministic given its seed (0, 1, 2). The JSONL files in `results/` already contain all runs, so the analysis and figure scripts work without re-running experiments.

## Using DyNeC on your own data

```python
import sys; sys.path.insert(0, "dynec")
import core
# Xtr, Xte: standardised float32 arrays; y_noisy: observed labels; yte: test labels
acc, aux = core.run((Xtr, y_noisy, Xte, yte, y_noisy), "dynec", epochs=25, warm=5, bs=128,
                    sizes=[Xtr.shape[1], 256, 128, n_classes], seed=0, C=n_classes, ret_labels=True)
corrected_labels = aux["labels"]       # arg max of the consensus soft targets
```
(The fifth element of the data tuple is used only for diagnostics; pass the noisy labels when true labels are unknown and do not request `log_auc`.)

## Default hyper-parameters

k = 10 neighbours, graph refresh every R = 3 epochs, temporal-ensemble coefficient 0.7, sharpening temperature 0.5, warm-up 5 epochs (10 for UCI Digits).

## Citation

See `CITATION.cff`. Please cite the article once published.

## License

MIT — see `LICENSE`.
