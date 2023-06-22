# Diffusion Recommender Model - Reproduction and Extension
This is a reproduction and extension of the paper at SIGIR 2023:
> [Diffusion Recommender Model](https://arxiv.org/abs/2304.04971)
> 
> Wenjie Wang, Yiyan Xu, Fuli Feng, Xinyu Lin, Xiangnan He, Tat-Seng Chua

Our reproduction is based on the [original code](https://github.com/YiyanXu/DiffRec). We extend the original code to include the following features:
- **WandB** logging
- **learnable temporal weighting** feature and visualization
- more suitable **early stopping** with increased patience
- additional **clustering** algorithms for `L-DiffRec` and `LT-DiffRec`

More details can be found in the [report](report.pdf), for a quick overview see the [poster](poster.pdf).


## Getting Started
We provide our working conda environment in the `environment.yml` file and add instructions to obtain the trained models.

1. Clone this repo if not already done so.
2. set up environment using conda and the provided environment.yml file:
    ```bash
    conda create --name rs --file environment.yml
    ```
3. activate the environment:
    ```bash
    source activate rs
    ```
4. Download the checkpoints released on [Google drive](https://drive.google.com/file/d/1bPnjO-EzIygjuvloLCGpqfBVnkrdG4IH/view?usp=share_link).
    ```bash
    gdown --id 1bPnjO-EzIygjuvloLCGpqfBVnkrdG4IH
    unzip checkpoints.zip -d checkpoints
    rm checkpoints.zip
    ```
4. set up [WandB](https://wandb.ai) logging: 
    
    We use  to log the training process. To use it, you need to create a new project in WandB and get your API key. Then run the following command, and provide your API key:
    ```bash
    python3 -m wandb login
    ```


## Usage 
### WandB Logging
- project name (`entity`) needs to be adapted in `main.py` for each model in the `wandb.init()` function.
- `--run_name`: set flag in order to add a description to the run name, e.g. `--run_name=TEST`


### Data
The experimental data are in the 'datasets' folder, and include Amazon-Book, Yelp and MovieLens-1M. 

#### Checkpoints
The checkpoints released by the authors are in './checkpoints' folder. To download them follow instructions in the Getting Started section.

### Run our Modifications
We use argparse flags to modify the hyperparameters of the models:
- modify early stopping **patience**: add flag `--patience` to set the number of epochs to wait before early stopping, default is 20.
- use learnable **temporal weighting** feature: add flag `--mean_type=x0_learnable` to use learnable temporal weighting feature.
- visualize the weights using m-PHATE: add flags `--visualize_weights` and `--mean_type=x0_learnable`. The generated graphs are saved on WandB. Alternatively, it is also possible to use the saved numpy array in the mPHATE folder to generate the graphs locally, using the `visualize_weights.py` script. 

    `--workers` specifies the number of parallel processes to use for the mPHATE algorithm.
    ```bash
    python utils/visualize_weights.py --dataset=amazon-book_clean --run_name=learnable_weight_test --model_type=T-DiffRec --seed 1 --workers 10
    ```

For `L-DiffRec` and `LT-DiffRec`:
- modify **cluserting algorithm**: add flag `--clustering_method=kmeans' to use default clustering method `kmeans`. Other implementations are `hierarchical`, `gmm`.
-- modify the amount of clusters used: add flag `--n_cate 2` to set the number of clusters, default is 2.


### Run the Reproduction
The reproductions can be run using the provided `run.job` files, or the following commands. Default values are chosen to match the original paper's choices.

#### DiffRec
...

#### L-DiffRec
...

#### T-DiffRec
...

#### LT-DiffRec
...

### Run Inference Only
We assume checkpoints have been downloaded and are in the `checkpoints` folder. Otherwise follow instructions in the Getting Started section.

#### Example Usage for Inference

```
python inference.py --dataset=$1 --gpu=$2
```

