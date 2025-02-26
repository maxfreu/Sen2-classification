import yaml
from experiments.train_and_validate import train_and_validate, load_data


logdir = "output"
experiment_name = "embedding_test"


with open(f"configs/statistics_223_g-5k.yaml") as f:
    norm_config = yaml.safe_load(f)["data"]

for embedding_type in (None, "bert", "concat"):
    for embedding_dim in (None, 32, 64, 128, 256):
        # skip combinations that make no sense
        if (embedding_type is None and embedding_dim is not None) or (embedding_type is not None and embedding_dim is None):
            continue
        data, dataconfig = load_data(data_args={"mean": norm_config["mean"], "stddev": norm_config["stddev"]})

        version = f"embedding_type={embedding_type}_embedding_dim={embedding_dim}_seq_len={dataconfig['sequence_length']}"

        train_and_validate("configs/gru.yaml",
                           data,
                           dataconfig | {"normalization": "223_g-5k"},
                           logdir,
                           experiment_name=experiment_name,
                           version=version,
                           experiment_file=__file__,
                           model_extra_args={"num_classes": data.num_classes,
                                             "classes": data.classes,
                                             "loss_weights": data.loss_weights,
                                             "embedding_type": embedding_type,
                                             "embedding_dim": embedding_dim},
                           )
