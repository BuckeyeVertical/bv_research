steps to set up:

```bash
module load miniconda3/24.1.2-py310
conda create -n .venv python=3.11 -y
conda activate .venv
pip install -r requirements.txt
```
make sure that you have wandb auth set up.

you can quickly set slurm directives by doing
```bash
export <slurm name>="<value>"

```


