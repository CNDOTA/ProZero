<h1 align="center">ProZero</h1>

<p align="center"><strong>Multi-objective protein directed evolution with pretrained protein language models and tree search</strong></p>

<p align="center">
  <img src="docs/prozero-badges.png" width="376" alt="Python 3.10, PyTorch 2.2.2, four benchmarks">
</p>

ProZero uses pretrained protein language models to propose mutations. Pareto
Monte Carlo tree search uses feedback from property predictors to explore
trade-offs across multiple design objectives. This repository contains the
search implementations for the four benchmarks described in the manuscript.

<p align="center">
  <img src="docs/prozero-framework.png" width="740" alt="ProZero workflow showing how protein language model priors and property scores guide Pareto tree search">
</p>

## 1. Benchmarks

### 1.0 Installation

Clone the repository once, then follow the setup instructions for the task you
want to run. Each task uses a separate Python 3.10 environment. The search
requires an NVIDIA GPU and a CUDA-enabled PyTorch installation.

```bash
git clone https://github.com/CNDOTA/ProZero.git
cd ProZero
```

After installing a task's requirements, check CUDA from its active environment:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

The CUDA check should print `True`. If it prints `False`, check your NVIDIA
driver and the [CUDA wheels for PyTorch 2.2.2](https://pytorch.org/get-started/previous-versions/).

| Task | Optimization objectives | Python requirements | Setup and run |
| --- | --- | --- | --- |
| 1.1 avGFP | Fluorescence and folding stability | [requirements-avgfp.txt](requirements-avgfp.txt) | [avGFP and low-N](#11-avgfp) |
| 1.2 PAB1 | RNA-binding score and folding stability | [requirements-pab1.txt](requirements-pab1.txt) | [PAB1](#12-pab1) |
| 1.3&nbsp;GCGR&#8288;/&#8288;GLP&#8288;-&#8288;1R | Predicted potency at both receptors | [requirements-gcgr-glp1r.txt](requirements-gcgr-glp1r.txt) | [Dual agonist design](#13-gcgrglp-1r) |
| 1.4 pHLA–TCR | Peptide–HLA binding, peptide–TCR recognition, presentation and pHLA stability | [requirements-phla-tcr.txt](requirements-phla-tcr.txt) | [Epitope design](#14-phla-tcr) |

### 1.1 avGFP

Both avGFP implementations optimize predicted fluorescence and an additive
Mutate Everything stability criterion.

#### 1.1.1 Install

From the repository root, create the environment for both avGFP runs.

```bash
conda create -n prozero-avgfp python=3.10 pip
conda activate prozero-avgfp
python -m pip install -r requirements-avgfp.txt
```

#### 1.1.2 Main benchmark

Follow the [third-party setup instructions](THIRD_PARTY.md)
to download the TAPE checkpoint and place it at the path below.
Both avGFP runs load the bundled
[mutation lookup](prozero/avgfp/assets/avgfp_ddg_lookup.csv) automatically.

```text
prozero/avgfp/tape_landscape/avGFP/
```

```bash
cd prozero/avgfp
python train.py --idx 1 --method prozero --gpus 0
```

The run writes a CSV to `prozero/avgfp/results/`. The search fixes the initial
methionine and allows edits at the remaining 237 residues with a trust radius
of 100. To run all five
benchmark trials, use `python batch_run.py --gpus 0` in `prozero/avgfp/`.

#### 1.1.3 Low-N benchmark

This run requires the fine-tuned ESM-1b landscape files.
Place or link the following assets under
`prozero/avgfp/low_n/`.

```text
esm1b_landscape/esm_params/
esm1b_landscape/avGFP/decoder.pt
esm1b_landscape/avGFP/starting_sequence.json
```

Set `PROZERO_ESM1B_LANDSCAPE` if these landscape files are stored elsewhere.
From the repository root:

```bash
cd prozero/avgfp/low_n
python train.py --idx 1 --method prozero --gpus 0
```

The run writes a CSV ending in `lown_r7_81aa.csv` under
`prozero/avgfp/low_n/results/`. It restricts edits to positions 29–109 with a
maximum depth of seven. In `prozero/avgfp/low_n/`, use
`python batch_run.py --gpus 0` for the 20-trial configuration.

### 1.2 PAB1

The PAB1 implementation optimizes the TAPE-predicted functional score together
with an additive Mutate Everything stability criterion. It uses ESM-1b to guide
mutation proposals and a multi-output value network in Pareto MCTS.

#### 1.2.1 Install

Run these commands from the repository root.

```bash
conda create -n prozero-pab1 python=3.10 pip
conda activate prozero-pab1
python -m pip install -r requirements-pab1.txt
```

#### 1.2.2 Prepare assets

Follow the [third-party setup instructions](THIRD_PARTY.md)
to download the TAPE checkpoint and place it at the path below. The PAB1 search
loads the bundled [mutation lookup](prozero/pab1/assets/pab1_ddg_lookup.csv)
automatically.

```text
prozero/pab1/tape_landscape/Pab1/
```

#### 1.2.3 Run

From the repository root:

```bash
cd prozero/pab1
python train.py --idx 1 --method prozero --gpus 0
```

The run writes a CSV to `prozero/pab1/results/`. To run all ten benchmark trials,
use `python batch_run.py --gpus 0` in `prozero/pab1/`.

### 1.3 GCGR/GLP-1R

This implementation optimizes the two predicted receptor potency objectives
from PeptideModels.

#### 1.3.1 Install

Run these commands from the repository root.

```bash
conda create -n prozero-gcgr-glp1r python=3.10 pip
conda activate prozero-gcgr-glp1r
python -m pip install -r requirements-gcgr-glp1r.txt
```

#### 1.3.2 Prepare assets

Obtain [PeptideModels](https://github.com/amp91/PeptideModels)
and place its importable package and trained models at the following paths.
The potency ensemble runs on CPU. The ESM-guided search and value network use CUDA.

```text
prozero/gcgr_glp1r/PeptideModels/peptide_models/
prozero/gcgr_glp1r/PeptideModels/peptide_models/models/
```

#### 1.3.3 Run

From the repository root:

```bash
cd prozero/gcgr_glp1r
python train.py --idx 1 --method prozero --gpus 0 --cpuct 10
```

The run writes a CSV ending in `c_puct10.csv` under `prozero/gcgr_glp1r/results/`.
In `prozero/gcgr_glp1r/`, use `python batch_run.py --gpus 0` for the five-trial
configuration with `cpuct=10`.

### 1.4 pHLA-TCR

The pHLA-TCR implementation optimizes Bertrand, NetMHCpan, MHCflurry and
NetMHCstab scores, with TULIP providing TCR- and HLA-conditioned mutation
guidance.

#### 1.4.1 Install

Run these commands from the repository root.

```bash
conda create -n prozero-phla-tcr python=3.10 pip
conda activate prozero-phla-tcr
python -m pip install -r requirements-phla-tcr.txt
mhcflurry-downloads fetch
```

#### 1.4.2 Prepare assets

Obtain the BERTrand package and checkpoint, TULIP source
and weights, and the standalone DTU tools. Place them inside `prozero/phla_tcr/`.
[Download and setup instructions](THIRD_PARTY.md#phlatcr-setup).

```text
bertrand/best_checkpoint/
bertrand/bertrand/...
TULIP/configs/shallow.config.json
TULIP/model_weights/pytorch_model.bin
TULIP/aatok/
TULIP/mhctok/
TULIP/src/...
netMHCpan-4.1/netMHCpan
netMHCstabpan-1.0/netMHCstabpan
```

#### 1.4.3 Run

From the repository root:

```bash
cd prozero/phla_tcr
python train.py \
  --idx 1 \
  --method prozero \
  --gpus 0 \
  --start_peptide ALIHHNTHL \
  --hla 'HLA-A*02:01' \
  --cdr3a CAVNSNSGYALNF \
  --cdr3b CASSQSETGDGYTF
```

The default HLA sequence corresponds to HLA-A*02:01. Supply `--hla_seq` when
using another allele. The run writes a CSV to `prozero/phla_tcr/results/`. In
`prozero/phla_tcr/`, use `python batch_run.py --gpus 0` to run the 27 manuscript
tasks. Use `--start-task` and `--end-task` to run a contiguous subset.

## 2. Reproducibility status

The manuscript data package contains figure-generation scripts and source data.
See the [reproducibility checklist](RELEASE_CHECKLIST.md) for the remaining
verification steps.

## 3. Citation

If you use ProZero in your research, please cite the accompanying manuscript.
