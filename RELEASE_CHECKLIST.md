# Public release checklist

## Required before claiming full manuscript reproduction

- Confirm the restored unrestricted avGFP settings and the separate low-N settings against the exact reported launch records.
- Confirm that the GCGR/GLP-1R entry point, output naming and cpuct=10 match the five reported performance runs.
- Record the exact Python, CUDA, model and external-tool versions used for each task.
- Document licensed installation steps for NetMHCpan and NetMHCstabpan.
- Verify the TULIP and BERTrand upstream versions and their additional dependencies.

## Code quality and reproducibility

- Replace fixed relative asset paths with command-line options or a checked configuration file.
- Expose CPU/GPU device selection consistently; several historical modules assume CUDA.
- Replace shell pipelines used for external predictors with checked subprocess argument lists and robust output parsing.
- Decide whether equal Pareto score vectors should dominate one another; the historical implementation can produce an empty candidate front when vectors are identical.
- Confirm that all manuscript hyperparameters match the final code and configuration files.

## Repository metadata

- Add the final citation, DOI, authors and contact details.
- Archive the release on Zenodo or an equivalent repository and cite the versioned DOI.
- Link the separate source-data and figure-code archive from the README and Data Availability statement.
