# Disruption

This repository contains the code for the embedding disruptiveness measure (EDM). You can find an extended abstract for IC2S2 here: <https://munjungkim.github.io/files/ic2s2_2023_03_02_disruptiveness.pdf>



# pip package

We developed a pip package for calculating the embedding disurptiveness measure. You can install it by 


```sh
pip install embedding-disruptiveness
```

The source code of the package is in [here](https://github.com/MunjungKim/embedding-disruptiveness), and the example code for calculating embedding disruptiveness measure with the package is in [`notebooks/embedding-disruptiveness package.ipynb`](https://github.com/yy/embedding-disruptiveness/blob/main/notebooks/embedding-disruptiveness%20package.ipynb).




# Environment setup

## uv

If you are using [uv](https://github.com/astral-sh/uv), run the following to install necessary packages:

```sh
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

# Snakemake

Since the current python code requires many parameters, it would be better to use snakemake tool. Once a container has been created, you can execute the following commands in the `workflow` directory:

```
snakemake '{working_dirctory}/data/original/150_5_q_1_ep_5_bs_4096_embedding/distance.npy' -j
```

This command will locate the rule for generating the `{working_directory}/data/original/150_5_q_1_ep_5_bs_4096_embedding/distance.npy` file, which corresponds to the `calculating_distance` rule in our Snakefile. 

This rule takes the `{working_directory}/data/original/150_5_q_1_ep_5_bs_4096_embedding/in.npy` and `{working_directory}/data/original/150_5_q_1_ep_5_bs_4096_embedding/out.npy` files as inputs, and Snakemake will execute the rule responsible for generating these files, named as the `embedding_all_network` rule in our Snakefile. 

The `embedding_all_network` rule executes the following command: `python3 scripts/Embedding.py {input} {params.Dsize} {params.Window} {params.Device1} {params.Device2} {params.Name} {params.q} {params.epoch} {params.batch}`. The parameters for this command are defined within the embedding_all_network rule as follows:

```

 params:
        Name = "{network_name}",
        Dsize = "{dim}",
        Window = "{win}",
        Device1 = "6",
        Device2 = "7",
        q = "{q}",
        epoch = "{ep}",
        batch = "{bs}"

```



# Without Snakemake

Without snakemake, you can follow the following steps.

## Embedding Calculation

To calculate the embedding vectors, you can use the following command line:

```
python3 scripts/Embedding.py {path/to/citation_network_file} {embedding_dimension} {window_size} {device1} {device2} {citation_network_name} {q_value} {epoch_size} {batch_size}
```

For example, you can run the command as shown below:

```
python3 scripts/Embedding.py /data/original/citation_net.npz 200 5 6 7 original 1 5 1024

```

`Embedding.py` will train the node2vec model and save the result of in-vectors under the path `{path/to/citation_network_file}/{DIM}_{WIN}_q_{Q}_ep_{EPOCH}_bs_{BATCH}_embedding/`. For instance, the above command will save in and out vectors in the path `/data/original/200_5_q_1_ep_5_bs_1024_embedding/in.npy` and `/data/original/200_5_q_1_ep_5_bs_1024_embedding/out.npy`.


## Distance Calculation

Based on the embedding vectors you calculate from the above process, you can execute the following command to calculate the distance. 

```
python3 scripts/Distance_Disruption.py distance {path/to/invectors}  {path/to/outvectors} {path/to/citation_network_file} {device name}
```

For example, you can run the command as shown below:

```
python3 scripts/Distance_Disruption.py distance /data/original/  {path/to/outvectors} {path/to/citation_network_file} {device name}
```
