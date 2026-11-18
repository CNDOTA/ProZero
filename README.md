# ProZero

**Multi-objective protein directed evolution with pretrained protein language models and tree search**

ProZero uses pretrained protein language models to propose mutations. Pareto
Monte Carlo tree search uses feedback from property predictors to explore
trade-offs across multiple design objectives. This repository contains the
search implementations for the four benchmarks described in the manuscript.

![ProZero workflow, showing how protein language model priors and property scores guide Pareto tree search](docs/prozero-framework.png)

## 1. Benchmarks

### Installation

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
| avGFP | Fluorescence and folding stability | [requirements-avgfp.txt](requirements-avgfp.txt) | [avGFP and low-N](#avgfp) |
| PAB1 | RNA-binding score and folding stability | [requirements-pab1.txt](requirements-pab1.txt) | [PAB1](#pab1) |
| GCGR/GLP-1R | Predicted potency at both receptors | [requirements-gcgr-glp1r.txt](requirements-gcgr-glp1r.txt) | [Dual agonist design](#gcgrglp-1r) |
| pHLA–TCR | Peptide–HLA binding, peptide–TCR recognition, presentation and pHLA stability | [requirements-phla-tcr.txt](requirements-phla-tcr.txt) | [Epitope design](#phla-tcr) |

### avGFP

Both avGFP implementations optimize predicted fluorescence and an additive
Mutate Everything stability criterion.

**Install.** From the repository root, create the environment for both avGFP runs.

```bash
conda create -n prozero-avgfp python=3.10 pip
conda activate prozero-avgfp
python -m pip install -r requirements-avgfp.txt
```

**Main benchmark.** Obtain the TAPE checkpoint and prepare the avGFP mutation
lookup at the following paths. [Checkpoint and lookup sources](THIRD_PARTY.md).

```text
prozero/avgfp/tape_landscape/avGFP/
prozero/avgfp/MutateEverything/example/avgfp_esm/avgfp_mut_dict.pkl
```

```bash
cd prozero/avgfp
python train.py --idx 1 --method prozero --gpus 0
```

The run writes a CSV to `prozero/avgfp/results/`. The search fixes the initial
methionine and allows edits at the remaining 237 residues with a trust radius
of 100. To run all five
benchmark trials, use `python batch_run.py --gpus 0` in `prozero/avgfp/`.

**Low-N benchmark.** In addition to the same mutation lookup, this run requires
the fine-tuned ESM-1b landscape files. Place or link the following assets under
`prozero/avgfp/low_n/`.

```text
MutateEverything/example/avgfp_esm/avgfp_mut_dict.pkl
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

### PAB1

The PAB1 implementation optimizes the TAPE-predicted functional score together
with an additive Mutate Everything stability criterion. It uses ESM-1b to guide
mutation proposals and a multi-output value network in Pareto MCTS.

**Install.** Run these commands from the repository root.

```bash
conda create -n prozero-pab1 python=3.10 pip
conda activate prozero-pab1
python -m pip install -r requirements-pab1.txt
```

**Prepare assets.** Place the TAPE checkpoint and mutation lookup inside
`prozero/pab1/`. [Checkpoint and lookup sources](THIRD_PARTY.md).

```text
tape_landscape/Pab1/
MutateEverything/example/pab1_esm/pab1_mut_dict.pkl
```

**Run.** From the repository root:

```bash
cd prozero/pab1
python train.py --idx 1 --method prozero --gpus 0
```

The run writes a CSV to `prozero/pab1/results/`. To run all ten benchmark trials,
use `python batch_run.py --gpus 0` in `prozero/pab1/`.

### GCGR/GLP-1R

This implementation optimizes the two predicted receptor potency objectives
from PeptideModels.

**Install.** Run these commands from the repository root.

```bash
conda create -n prozero-gcgr-glp1r python=3.10 pip
conda activate prozero-gcgr-glp1r
python -m pip install -r requirements-gcgr-glp1r.txt
```

**Prepare assets.** Obtain [PeptideModels](https://github.com/amp91/PeptideModels)
and place its importable package and trained models at the following paths.
The potency ensemble runs on CPU. The ESM-guided search and value network use CUDA.

```text
prozero/gcgr_glp1r/PeptideModels/peptide_models/
prozero/gcgr_glp1r/PeptideModels/peptide_models/models/
```

**Run.** From the repository root:

```bash
cd prozero/gcgr_glp1r
python train.py --idx 1 --method prozero --gpus 0 --cpuct 10
```

The run writes a CSV ending in `c_puct10.csv` under `prozero/gcgr_glp1r/results/`.
In `prozero/gcgr_glp1r/`, use `python batch_run.py --gpus 0` for the five-trial
configuration with `cpuct=10`.

### pHLA-TCR

The pHLA-TCR implementation optimizes Bertrand, NetMHCpan, MHCflurry and
NetMHCstab scores, with TULIP providing TCR- and HLA-conditioned mutation
guidance.

**Install.** Run these commands from the repository root.

```bash
conda create -n prozero-phla-tcr python=3.10 pip
conda activate prozero-phla-tcr
python -m pip install -r requirements-phla-tcr.txt
mhcflurry-downloads fetch
```

**Prepare assets.** Obtain the BERTrand package and checkpoint, TULIP source
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

**Run.** From the repository root:

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
