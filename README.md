# DropGNN
This is the official implementation of [DropGNN: Random Dropouts Increase the Expressiveness of Graph Neural Networks](https://arxiv.org/abs/2111.06283)
as described in the following NeurIPS 2021 paper:
```
@InProceedings{papp2021dropgnn,
  title={DropGNN: Random Dropouts Increase the Expressiveness of Graph Neural Networks}, 
  author={Papp, P{\'a}l Andr{\'a}s and Martinkus, Karolis and Faber, Lukas and Wattenhofer, Roger}, 
  booktitle={35th Conference on Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```

## Dependencies
The implementation uses Python (3.8), PyTorch (1.9) and PyTorch Geometric (1.7).
You can create a conda environment with all of the required packages:  
`conda env create -f environment.yml`  
`conda activate DropGNN`

## Runing The Experiments

Each experiment and base model are in an individual Python file and can be run individaully from the command line.

### Synthetic Benchmarks
Synthetic datasets and the GIN model with the corresponding augmentations is implemented in `gin-synthetic.py`
You can run experiments on the dataset you whish `['skipcircles', 'triangles', 'lcc', 'limitsone', 'limitstwo', 'fourcycles']` using either of the augmentations `['none', 'ports', 'ids', 'random', 'dropout']`:

`python gin-synthetic.py --augmentation 'dropout' --use_aux_loss --dataset 'lcc'`

DropGNN model version (`dropout` augmentation) should be run with `--use_aux_loss` flag, for other augmentations this option is ignored. For SkipCircles (`skipcircles`) dataset we use `--num_layers 9` flag with all augmentations.

The number of runs and dropout probability ablations can be run using the following commands:

`python gin-synthetic.py --augmentation 'dropout' --use_aux_loss --dataset 'limitsone' --num_runs_ablation`

`python gin-synthetic.py --augmentation 'dropout' --use_aux_loss --dataset 'limitsone' --prob_ablation`

### Graph Classification
Graph classification experiment code can be found in `gin-graph_classification.py`. You can run the DropGIN model on one of the datasets `['MUTAG', 'PTC', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI']` using the following command:

`python gin-graph_classification.py --drop_gnn --use_aux_loss --dataset 'MUTAG'`

To perform the grid search as done in the [orginal GIN paper](https://arxiv.org/abs/1810.00826) add `--grid_search` option. To run the grid search on SLURM cluster using GPUs use `--slurm` and `--gpu_jobs` options. The code is currently setup for grid search on the bio datasets. To run grid search on social IMDB datasets as described in the paper (with fixed number of hidden units) set `tunable=False` for the `hidden_units` option on line 323.
 
### Graph Property Regression
Code for QM9 experiment with DropMPNN and Drop-1-GNN models can be respectively found in `mpnn-qm9.py` and `1-gnn-qm9.py` files.
You can run the DropGNN experiments using the following commands:

`python mpnn-qm9.py --drop_gnn --aux_loss --target 0`

`python 1-gnn-qm9.py --drop_gnn --aux_loss --target 0`

The `--target` flag specifies which of the 12 graph properties the model should fit.

Note that the code reports energy values in eV, while in the paper they are converted to Hartree.
