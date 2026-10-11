# MosqEdits Full Pipeline

MosqEdits predicts and ranks gRNA on-target activity from a target-region FASTA, searches a matching reference genome for potential off-target sites, evaluates the effects of population SNPs, and produces an integrated guide-level results table.

## Workflow

```text
Target-region FASTA
    ↓
Generate and rank candidate gRNAs by predicted on-target activity
    ↓
Search the reference genome for potential off-target sites
    ↓
Evaluate population SNPs and score off-target candidates
    ↓
Aggregate guide-level risk and combine it with on-target predictions
```

The repository is organized around two components:

- `on_target/`: candidate generation, on-target prediction, and model files.
- `off_target/`: the 01–10 off-target analysis steps, prediction models, reference data, and tools.

`run_complete.py` connects both components. The example target region is in `examples/white_AalbF3_region.fa`.

## Installation

The pipeline is intended for Linux and uses **two separate Conda environments** because the on-target and off-target stages have different dependency stacks. Both specifications are stored in `env/`; a standard Conda environment file defines one environment, so create each separately from the repository root:

```bash
conda env create --file env/environment.on_target.yml
conda env create --file env/environment.off_target.yml
```

`env/environment.on_target.yml` creates `crispron` for on-target prediction (TensorFlow and `RNAfold`); `env/environment.off_target.yml` creates `mosqguide-off` for off-target scoring (PyTorch, scikit-learn, and CPU OpenCL/PoCL). `run_complete.py` runs the on-target stage in `crispron` first, then starts the off-target stage in `mosqguide-off` with `conda run`. You can launch it from any active environment as long as both named environments exist. If your environment names differ, use `--on-target-env NAME` and `--off-target-env NAME`.

## Required downloads

### AalbF3 reference genome

The example uses the *Aedes albopictus* AalbF3 assembly **GCA_018104305.1**. Download its genomic FASTA from the [NCBI assembly page](https://www.ncbi.nlm.nih.gov/datasets/genome/GCA_018104305.1/). Extract the download and place the FASTA at:

```text
off_target/reference/AalbF3/GCA_018104305.1_AalbF3_genomic.fna
```

Rename the extracted FASTA if necessary to match this path. The genome is not stored in this GitHub repository.

### Random forest models

The two random forest models are distributed with [MosqGuide-Off](https://github.com/GeneEditing/MosqGuide-Off#data-and-model-files). Download each file and save it at the indicated path:

| Model | HTTP download address | Save as |
| --- | --- | --- |
| NGG random forest | `http://1.12.219.134/1s2ZOEQeaKN3KC9a7qNF/resources/MosqGuide-Off/pam_ngg/checkpoints/rf_handcrafted_baseline.joblib` | `off_target/pam_ngg/checkpoints/rf_handcrafted_baseline.joblib` |
| PAM-agnostic random forest | `http://1.12.219.134/1s2ZOEQeaKN3KC9a7qNF/resources/MosqGuide-Off/pam_agnostic/checkpoints/rf_handcrafted_baseline.joblib` | `off_target/pam_agnostic/checkpoints/rf_handcrafted_baseline.joblib` |

Copy the HTTP address into your browser's address bar to download the file. The SNP table and other smaller model files are included in this repository.

## Run the example

After placing the required files at the paths above, run the AalbF3 white target-region example from the repository root:

```bash
python run_complete.py --pam NGG
```

For a non-NGG PAM, such as NGTT:

```bash
python run_complete.py --pam NGTT
```

To analyze your own target region:

```bash
python run_complete.py --input path/to/target.fa --gene my_target --pam NGG
```

Run `python run_complete.py --help` for additional options, including custom genome and SNP files, output location, mismatch limit, and Conda environment names.

## Input and output

For the complete pipeline, the FASTA record ID should identify the reference contig and its **1-based inclusive** genomic coordinates:

```text
>contig:start-end
```

See `examples/white_AalbF3_region.fa` for an example. The target-region FASTA, reference genome, and SNP table must use matching genomic coordinates and assembly versions.

By default, results are written under `results/<gene>/`. The `on_target/` subdirectory contains ranked activity predictions; `final_guides_with_risk.tsv` combines these predictions with guide-level off-target risk.

## License and third-party components

Original MosqEdits code is released under the MIT License; see `LICENSE`. 

- **CRISPRon:** The NGG on-target implementation and pretrained models originate from [RTH-tools/CRISPRon](https://github.com/RTH-tools/crispron). CRISPRon is published under the GNU Affero General Public License, version 3 or later. Its original copyright and license notices are retained.
- **CRISPRoff/CRISPRspec:** `on_target/crispron_official/bin/CRISPRspec_CRISPRoff_pipeline.py` and its energy parameters originate from [RTH-tools/CRISPRoff](https://github.com/RTH-tools/crisproff). The script contains its own AGPL notice; the upstream repository also provides its license and author information. These upstream terms and notices remain applicable.
- **Cas-OFFinder:** The genome search uses [Cas-OFFinder](https://github.com/snugel/cas-offinder), which retains its upstream copyright and license terms.