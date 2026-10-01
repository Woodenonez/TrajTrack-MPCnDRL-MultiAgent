# DDPG training and checkpoint evaluation

`src/training.py` keeps the training functions and dataclasses; `scripts/train.py` is the command-line entry. This checkout implements DDPG with prioritized replay for two variants: index **0** uses image observations, `[64, 64]`, and CUDA; index **1** uses ray observations, `[16, 16]`, and CPU. Older TD3 and four-variant descriptions refer to historical code and are not active here.

From the repository root, after [setup](../../README.md#setup):

```bash
python scripts/train.py --mode local --index 1 --run-vers 0 --path Model/training/variant-1/run0
python scripts/train.py --config config/training.yaml --no-evaluation --index 1 --path Model/training/variant-1/run0
```

A dataclass-only launch trains by default. The supplied YAML has `evaluation: true`, so loading it without `--no-evaluation` attempts to load a checkpoint. Local training defaults to 100,000 total steps; cluster training defaults to 7,000,000. The training code uses 20 environments unless `--n-cpu` changes it. A short run at the default 100,000-step learning threshold does not establish that an optimizer update occurred.

To evaluate a **full Stable-Baselines3 archive** already saved by this trainer:

```bash
python scripts/train.py --mode cluster --index 1 --evaluation --evaluation-episodes 2 --no-render --path Model/training/variant-1/run0
```

`--path` names the run directory; the loader reads its `best_model` archive, and training writes `final_model` plus evaluation data there. These are distinct from the shipped `pretrained_model/*/best_model.pt` policy state dictionaries used by the demo runners. Do not supply a `.pt` file to training evaluation. Omitting `--evaluation-episodes` keeps local evaluation unlimited and cluster evaluation at one episode; `--no-render` only suppresses evaluation displays. Each episode retains its 1,000-step limit.

## SLURM template

[SLURM_jobscript.sh](SLURM_jobscript.sh) is a site-specific example, not a verified submission. Update its `#SBATCH` account, resources, and job name for your cluster. Set absolute paths for the checkout, Apptainer image, and **persistent** model output before submission, for example:

```bash
export REPO_PATH=/persistent/path/to/repository
export CONTAINER=/persistent/path/to/runtime.sif
export OUTPUT_ROOT=/persistent/path/to/training-runs
export MODEL=0 RUN=19
sbatch --export=ALL doc/training/SLURM_jobscript.sh
```

The template binds the repository and output directory into the container and passes the persistent run path through `--path`; outputs are not left in node-local temporary storage. `MODEL=0` needs CUDA. A local Python environment can be used instead of a container by adapting this template to your site; no cluster environment was tested in this checkout.
