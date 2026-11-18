# Third-party dependencies and assets

Start with the [Python environment and requirements](README.md#installation)
for your benchmark. Then obtain the assets for that task from the sources below
and place them at the listed paths, which are relative to the repository root.
Follow each provider's installation and licensing instructions.

| Component | Used by | Source | Expected local location or behavior |
| --- | --- | --- | --- |
| TAPE avGFP landscape checkpoint | avGFP fluorescence oracle | [Manuscript checkpoint archive](https://drive.google.com/file/d/1uy9zgtJ60Z83LCbm7Z_AoksAkmK4CcsC/view) | `prozero/avgfp/tape_landscape/avGFP/` |
| avGFP mutation lookup | avGFP main and low-N stability objective | [Bundled CSV](prozero/avgfp/assets/avgfp_ddg_lookup.csv), predicted with [Mutate Everything](https://github.com/jozhang97/MutateEverything) | Loaded automatically by both avGFP runs. |
| ESM-1b landscape decoder | avGFP low-N fluorescence oracle | Project-specific asset | Root selected by `PROZERO_ESM1B_LANDSCAPE`, containing `esm_params/`, `avGFP/decoder.pt` and `avGFP/starting_sequence.json` |
| ESM-1b / fair-esm | avGFP, PAB1 and GCGR/GLP-1R mutation guidance | [ESM-1b model card and weights](https://huggingface.co/facebook/esm1b_t33_650M_UR50S) | `fair-esm` downloads its checkpoint on first use unless already cached. |
| TAPE PAB1 landscape checkpoint | PAB1 oracle | [Manuscript checkpoint archive](https://drive.google.com/file/d/1uy9zgtJ60Z83LCbm7Z_AoksAkmK4CcsC/view) | `prozero/pab1/tape_landscape/Pab1/` |
| Mutate Everything PAB1 lookup | PAB1 stability objective | [Upstream code and weights](https://github.com/jozhang97/MutateEverything) | `prozero/pab1/MutateEverything/example/pab1_esm/pab1_mut_dict.pkl` |
| BERTrand | pHLA-TCR oracle | [Upstream code and checkpoint instructions](https://github.com/SFGLab/bertrand) | Python package tree under `prozero/phla_tcr/bertrand/` and checkpoint under `best_checkpoint/` |
| TULIP | pHLA-TCR mutation guidance | [Upstream code and model weights](https://github.com/barthelemymp/TULIP-TCR) | Source, tokenizers, config and weights under `prozero/phla_tcr/TULIP/` |
| NetMHCpan 4.1 | pHLA binding objective | [DTU software and Downloads tab](https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/) | `prozero/phla_tcr/netMHCpan-4.1/netMHCpan` |
| NetMHCstabpan 1.0 | pHLA stability objective | [DTU software and Downloads tab](https://services.healthtech.dtu.dk/services/NetMHCstabpan-1.0/) | `prozero/phla_tcr/netMHCstabpan-1.0/netMHCstabpan` |
| MHCflurry | Presentation objective | [Upstream installation and model download instructions](https://github.com/openvax/mhcflurry) | Install the version in [`requirements-phla-tcr.txt`](requirements-phla-tcr.txt) and download the model data. |
| PeptideModels | GCGR/GLP-1R potency objectives | [Upstream code and models](https://github.com/amp91/PeptideModels) | `prozero/gcgr_glp1r/PeptideModels/peptide_models/`, including `models/` |

The bundled avGFP lookup contains the single-site ΔΔG predictions used by the
benchmarks, in kcal/mol. Positions are zero-based in the search sequence, which
omits the fixed initial methionine. Prepare the PAB1 lookup at the path above.
The low-N search also needs its project-specific decoder and starting-sequence
file in addition to the base ESM-1b weights.

## pHLA–TCR setup

After installing `requirements-phla-tcr.txt`, fetch the MHCflurry model data in
the same environment:

```bash
mhcflurry-downloads fetch
```

Use the Downloads tabs on the two DTU pages to obtain the standalone NetMHCpan
4.1 and NetMHCstabpan 1.0 distributions. Follow the provider's setup instructions
and check that the executables are available at the paths in the table.

The requirements files cover direct Python imports in the staged code. BERTrand
and TULIP may impose additional dependencies, including TensorFlow or other
packages from their upstream installations.

This staging directory does not assign a license to third-party source code,
models or data. Do not copy those assets into the public repository unless their
licenses permit redistribution.

The avGFP low-N implementation requires the sequence_models package for its
fine-tuned ESM-1b decoder. The unrestricted TAPE configuration does not import
that optional oracle path.
