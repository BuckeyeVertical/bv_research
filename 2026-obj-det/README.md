steps to set up:

```bash
module load miniconda3/24.1.2-py310

conda env create -f environment.yml

conda activate .venv
```
make sure that you have wandb auth set up.

you can quickly set slurm directives by doing
```bash
export <slurm name>=<value>

```


