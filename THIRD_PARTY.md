# Third-party dependencies and assets

The following components are required by the task code but are not bundled.
Users must obtain them from their official sources and comply with their terms.

| Component | Used by | Expected local location or behavior |
| --- | --- | --- |
| TAPE avGFP landscape checkpoint | avGFP fluorescence oracle | prozero/avgfp/tape_landscape/avGFP/ |
| Mutate Everything avGFP lookup | avGFP stability objective | prozero/avgfp/MutateEverything/example/avgfp_esm/avgfp_mut_dict.pkl |
| ESM1b landscape decoder | avGFP low-N fluorescence oracle | Root selected by PROZERO_ESM1B_LANDSCAPE; contains esm_params/ and avGFP/decoder.pt |
| ESM-1b / fair-esm | PAB1 mutation guidance | Downloaded by fair-esm on first use unless already cached. |
| TAPE PAB1 landscape checkpoint | PAB1 oracle | `prozero/pab1/tape_landscape/Pab1/` |
| Mutate Everything PAB1 lookup | PAB1 stability objective | `prozero/pab1/MutateEverything/example/pab1_esm/pab1_mut_dict.pkl` |
| BERTrand | pHLA-TCR oracle | Python package tree under `prozero/phla_tcr/bertrand/` and checkpoint under `best_checkpoint/` |
| TULIP | pHLA-TCR mutation guidance | Source, tokenizers, config and weights under `prozero/phla_tcr/TULIP/` |
| NetMHCpan 4.1 | pHLA binding objective | `prozero/phla_tcr/netMHCpan-4.1/netMHCpan` |
| NetMHCstabpan 1.0 | pHLA stability objective | `prozero/phla_tcr/netMHCstabpan-1.0/netMHCstabpan` |
| MHCflurry 2.0 | Presentation objective | Installed Python package and downloaded model data |
| PeptideModels | GCGR/GLP-1R potency objectives | prozero/gcgr_glp1r/PeptideModels/peptide_models/, including trained models/ |

The requirements files cover direct Python imports in the staged code. BERTrand
and TULIP may impose additional dependencies, including TensorFlow or other
packages from their upstream installations.

This staging directory does not assign a license to third-party source code,
models or data. Do not copy those assets into the public repository unless their
licenses permit redistribution.

The avGFP low-N implementation requires the sequence_models package for its
fine-tuned ESM-1b decoder. The unrestricted TAPE configuration does not import
that optional oracle path.
