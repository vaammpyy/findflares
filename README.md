# findflares

A calibrated stellar flare detection pipeline for TESS.

## Environment and Requirements

### Conda

Create a conda environment.

`conda create -n <env-name> python=3.11`

Activate the environment.

`conda activate <env-name>`

Before installing the dependencies upgrade the pip.

`pip install -U pip`

Installing the dependencies.

`pip install celerite2 pymc-ext "exoplanet[pymc]" lightkurve numpy matplotlib scipy astropy`

---

## Pipeline execution

Runs the pipeline for all the TESS sectors with `20, 120` cadence.
`python3 run_pipeline.py -t <TIC>`

Re-runs the pipeline for all the TESS sectors with `20, 120` cadence.
`python3 run_pipeline.py -t <TIC> -r`

Runs the pipeline for all the given TESS sectors and cadence.
`python3 run_pipeline.py -t <TIC> -s <sector> -c <cadence>`

Run the pipeline for all the TESS sectors with `20, 120` cadence with injection recovery tests.
`python3 run_pipeline.py -t <TIC> -i <number-of-injrec-tests>`

Work in progress.
