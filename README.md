# ProZero

ProZero is a protein language model-guided Pareto Monte Carlo tree search
framework for multi-objective protein directed evolution. This repository
stages the core search code used for the four tasks described in the
accompanying manuscript.

## Repository contents

```text
prozero/
  avgfp/         unrestricted avGFP benchmark search
    low_n/       avGFP low-N search
  pab1/          PAB1 search implementation
  gcgr_glp1r/    GCGR/GLP-1R search implementation
  phla_tcr/      pHLA-TCR search implementation
requirements-avgfp.txt
requirements-pab1.txt
requirements-gcgr-glp1r.txt
requirements-phla-tcr.txt
THIRD_PARTY.md
RELEASE_CHECKLIST.md
```

Figure-generation code and source data are maintained with the manuscript data
package and are intentionally not duplicated here. Generated results, model
weights, licensed predictors and wet-laboratory data are excluded.

## avGFP

Both avGFP implementations optimize predicted fluorescence and an additive
Mutate Everything stability criterion. Install requirements-avgfp.txt.

The unrestricted benchmark uses the TAPE checkpoint and mutation lookup at:

    prozero/avgfp/tape_landscape/avGFP/
    prozero/avgfp/MutateEverything/example/avgfp_esm/avgfp_mut_dict.pkl

The low-N search uses the same mutation lookup and a fine-tuned ESM-1b decoder.
From prozero/avgfp/low_n, provide or link:

    MutateEverything/example/avgfp_esm/avgfp_mut_dict.pkl
    esm1b_landscape/esm_params/
    esm1b_landscape/avGFP/decoder.pt
    esm1b_landscape/avGFP/starting_sequence.json

Set PROZERO_ESM1B_LANDSCAPE if the ESM landscape root is stored elsewhere.

Run the unrestricted main benchmark search:

    cd prozero/avgfp
    python train.py --idx 1 --method prozero --gpus 0

The full avGFP sequence has 238 residues. Its initial methionine is fixed and
omitted from the search state; this searches all remaining 237 designable
residues with a maximum mutation depth of 100 and writes the main benchmark
filename.

Run the low-N search separately:

    cd prozero/avgfp/low_n
    python train.py --idx 1 --method prozero --gpus 0

The low-N implementation limits mutations to positions 29-109 and sets the
maximum mutation depth to seven. It writes a filename ending in
lown_r7_81aa.csv.

The unrestricted batch launcher defaults to five runs; the low-N batch launcher
defaults to twenty runs:

    cd prozero/avgfp
    python batch_run.py --gpus 0

    cd low_n
    python batch_run.py --gpus 0

## PAB1

The PAB1 implementation optimizes the TAPE-predicted functional score together
with an additive Mutate Everything stability criterion. It uses ESM-1b to guide
mutation proposals and a multi-output value network in Pareto MCTS.

Install the Python dependencies in a Python 3.10 environment:

```bash
pip install -r requirements-pab1.txt
```

Before running, provide these external assets inside `prozero/pab1/`:

```text
tape_landscape/Pab1/
MutateEverything/example/pab1_esm/pab1_mut_dict.pkl
```

Run one search from the task directory:

```bash
cd prozero/pab1
python train.py --idx 1 --method prozero --gpus 0
```

The result is written to `results/`. The batch launcher defaults to the ten
independent runs reported in the manuscript:

    python batch_run.py --gpus 0

The code expects CUDA.

## GCGR/GLP-1R

This implementation optimizes the two predicted receptor potency objectives
from PeptideModels. Install requirements-gcgr-glp1r.txt. Obtain PeptideModels
separately and place its importable package and trained models at:

    prozero/gcgr_glp1r/PeptideModels/peptide_models/
    prozero/gcgr_glp1r/PeptideModels/peptide_models/models/

Run one search:

    cd prozero/gcgr_glp1r
    python train.py --idx 1 --method prozero --gpus 0 --cpuct 10

The potency ensemble is evaluated on CPU in the historical implementation;
ESM-guided search and the value network use CUDA. The public-release copy writes
the standard c_puct result filename rather than the uploaded timing-run name.

The batch launcher defaults to five runs with cpuct 10:

    python batch_run.py --gpus 0

## pHLA-TCR

The pHLA-TCR implementation optimizes Bertrand, NetMHCpan, MHCflurry and
NetMHCstab scores, with TULIP providing TCR- and HLA-conditioned mutation
guidance.

```bash
pip install -r requirements-phla-tcr.txt
```

The implementation also requires the separately obtained assets below inside
`prozero/phla_tcr/`:

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

Install the MHCflurry models using the instructions from the MHCflurry project.
NetMHCpan and NetMHCstabpan have their own licenses and are not redistributed.

Run one task from the task directory:

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
using another allele. The result is written to `results/`.

The batch launcher contains all 27 manuscript task definitions and runs each
task once:

    python batch_run.py --gpus 0

Use `--start-task` and `--end-task` to run a contiguous subset.

## Reproducibility status

This release includes the four uploaded task implementations. External model
assets and licensed predictors must be obtained separately as described above.
See `RELEASE_CHECKLIST.md` for the remaining code and documentation items.

## Citation

Please cite the ProZero manuscript. A final DOI and citation metadata should be
added before the public release.
