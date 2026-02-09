import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
import experiments.train_and_validate
from sen2classification import utils
from sen2classification.datasets import InMemoryTimeSeriesDataset
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sen2classification.models.gru import GRU
from torch import nn, optim
from torch.nn.functional import relu, softmax
from itertools import islice


color_dict = {i: random_color() for i in range(14)}

# thanks gpt
color_dict = {
    0: '#0cf5e5',  # AL (Alder)
    1: '#844a08',  # ASH (Ash)
    2: '#f0120e',  # BE (Beech)
    3: '#000000',  # BG (Background) - Slate gray, neutral or background species color
    4: '#a2ee14',  # BI (Birch)
    5: '#a404d0',  # DG (Douglas Fir)
    6: '#3ee295',  # FIR (Fir)
    7: '#f2f212',  # LA (Larch)
    8: '#ff01c8',  # MA (Maple)
    9: '#1853d2',  # OA (Oak)
    10: '#000000',  # OC (Other Coniferous)
    11: '#8b8b8b',  # OD (Other Deciduous)
    12: '#f1b913',  # PI (Pine)
    13: '#06ae0c',  # SP (Spruce)
}

# class_mapping = {
#     0: 'Erle',
#     1: 'Esche',
#     2: 'Buche',
#     3: 'Hintergr.',
#     4: 'Birke',
#     5: 'Douglasie',
#     6: 'Tanne',
#     7: 'Lärche',
#     8: 'Ahorn',
#     9: 'Eiche',
#     10: 'A. Konif.',
#     11: 'A. Laubb.',
#     12: 'Kiefer',
#     13: 'Fichte'
# }
class_mapping = {
    0: 'Alder',
    1: 'Ash',
    2: 'Beech',
    3: 'Backgr.',
    4: 'Birch',
    5: 'Douglas Fir',
    6: 'Fir',
    7: 'Larch',
    8: 'Maple',
    9: 'Oak',
    10: 'O. Conif.',
    11: 'Broadl.',
    12: 'Pine',
    13: 'Spruce'
}


def random_color():
    return np.random.choice(range(256), size=3) / 255


class IntermediateOutputModel(GRU):
    def __init__(self, model):
        super().__init__()
        self.embed_time = model.embed_time
        self.pos_embed = model.pos_embed
        self.norm = model.norm
        self.gru = model.gru
        self.fc1 = model.fc1
        self.clf = model.clf
        self.dropout = model.dropout

    def forward(self, boa, times, mask, validation_mask=None):
        z = self.norm(boa)

        if self.embed_time:
            z = self.pos_embed(z, times)

        z = nn.utils.rnn.pack_padded_sequence(z, torch.sum(torch.logical_not(mask), 1).cpu(), batch_first=True, enforce_sorted=False)
        z, h = self.gru(z)
        z, mask_lens = nn.utils.rnn.pad_packed_sequence(z, batch_first=True, total_length=mask.shape[1])
        z = z[torch.arange(len(mask)), mask_lens-1, :]
        z = self.dropout(z)
        z = self.fc1(z)
        z = relu(z)
        z = self.clf(z)
        return z


#%%
data_config_path = "output/final_cross_validation_GRU/cross_val=0/config.yaml"
model_config_path = data_config_path
ckpt = "/home/max/dr/Sen2-classification/output/final_cross_validation_GRU/cross_val=0/checkpoints/epoch=38-step=99411-val_loss=0.7419.ckpt"


with open(data_config_path, "r") as f:
    dataconfig = yaml.safe_load(f)["data"]

del dataconfig["normalization"]

class_mapping = {i:c for (i,c) in enumerate(list(sorted(set(dataconfig["class_mapping"].values()))))}

with open("configs/statistics_223_g-5k.yaml", "r") as f:
    normalization = yaml.safe_load(f)["data"]
    mean = np.array(normalization["mean"]).astype(np.float32)
    stddev = np.array(normalization["stddev"]).astype(np.float32)

model, _ = utils.load_model_from_configs_and_checkpoint(model_config_path, data_config_path, ckpt)
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
# model = torch.compile(model)

#%%
intermediate_model = IntermediateOutputModel(model)
#%%
# intermediate_model(boa,doy,mask).shape
#%%
# data = experiments.train_and_validate.load_data(overwrite_args=dataconfig | {"where": "tree_id < -70000"})
#%%
test_dataset = InMemoryTimeSeriesDataset(
    # input_filepath=dataconfig["input_file"],
    input_filepath="/home/max/dr/extract_sentinel_pixels/datasets/S2GNFI_V1.parquet",
    dbname=dataconfig["dbname"],
    sequence_length=dataconfig["sequence_length"],
    quality_mask=dataconfig["quality_mask"],
    class_mapping=dataconfig["class_mapping"],
    return_mode=dataconfig["return_mode"],
    time_encoding=dataconfig["time_encoding"],
    # plot_ids=dataconfig["val_ids"] if "val_ids" in dataconfig.keys() else None,
    plot_ids=dataconfig["val_ids"],
    # where="" if "plot_ids" in dataconfig.keys() else "is_train = FALSE",
    where="is_pure=TRUE",
    mean=np.array(dataconfig["mean"]),
    stddev=np.array(dataconfig["stddev"])
)


#%%
preds = []
trues = []
batchsize=32
dl = DataLoader(test_dataset, batch_size=batchsize)

with torch.no_grad():
    for (i, batch) in enumerate(dl):
        # if i >= 1:
        #     break

        if i%batchsize == 0:
            print(f"{i} / {len(dl)}")
        tree_ids, boa, doy, mask, y_true = batch
        # boa = boa.to(device)
        # doy = doy.to(device)
        # mask = mask.to(device)
        # y_pred = model(boa, doy, mask)
        y_pred = intermediate_model(boa, doy, mask)
        preds.extend(list(y_pred.numpy()))
        trues.extend(list(y_true.numpy()))
        # break

#%%
preds[0].shape

#%%
preds = np.array(preds)
trues = np.array(trues)
y_pred = preds.argmax(axis=1)

#%%
rand_selection = np.random.rand(len(preds)) > 0.2

selected_preds = preds[rand_selection]
selected_trues = trues[rand_selection]
selected_y_pred = y_pred[rand_selection]

#%%
# throw out background
bg = 3
conif = 10
selected_y_pred = selected_y_pred[np.logical_and(selected_trues != bg, selected_trues != conif)]
selected_preds  =  selected_preds[np.logical_and(selected_trues != bg, selected_trues != conif)]
selected_trues  =  selected_trues[np.logical_and(selected_trues != bg, selected_trues != conif)]
#%%
tsne = TSNE(
    n_components=2,
    init="pca",
    random_state=0,
    perplexity=50,
    learning_rate="auto",
)
Y = tsne.fit_transform(selected_preds)
#%%

def plot_tsne(Y, selected_trues, selected_y_pred, color_dict, class_mapping, alpha=0.1, filename="tsne_plot.pdf"):
    """
    Plot t-SNE embeddings with class-wise colors and prediction correctness.

    Parameters:
        Y (ndarray): 2D t-SNE coordinates (N x 2)
        selected_trues (ndarray): True labels for the selected points (length N)
        selected_y_pred (ndarray): Predicted labels for the selected points (length N)
        color_dict (dict): Mapping from class index to color
        class_mapping (dict): Mapping from class index to label string
        filename (str): Output filename (PDF)

    Returns:
        fig (matplotlib.figure.Figure): The created figure
    """

    fig, ax = plt.subplots(1, 2, figsize=(170 / 25.4, 80 / 25.4), constrained_layout=True)

    # LEFT: Class-wise plot
    # for i in reversed(range(14)):
    # for i in range(14):
    # for i in reversed((0,1,2,4,5,6,7,8,9,11,12,13)):
    for i in (0,1,2,4,5,6,7,8,9,11,12,13):
        tsne_pts = Y[selected_trues == i]
        ax[0].scatter(tsne_pts[:, 0], tsne_pts[:, 1],
                      color=color_dict[i], label=class_mapping[i], alpha=alpha, s=5)

    ax[0].set_aspect('equal')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    for spine in ax[0].spines.values():
        spine.set_visible(False)

    # Legend to the right of first plot (vertical)
    # leg0 = ax[0].legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
    #                     frameon=False, fontsize=6)
    # for lh in leg0.legend_handles:
    #     lh.set_alpha(1)

    # RIGHT: Correct vs incorrect
    false_pts = Y[selected_trues != selected_y_pred]
    true_pts = Y[selected_trues == selected_y_pred]
    ax[1].scatter(true_pts[:, 0], true_pts[:, 1],   color="blue", label="Right", alpha=alpha, s=5)
    ax[1].scatter(false_pts[:, 0], false_pts[:, 1], color="red",  label="Wrong", alpha=alpha, s=5)

    ax[1].set_aspect('equal')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    for spine in ax[1].spines.values():
        spine.set_visible(False)

    # leg1 = ax[1].legend(loc='lower center',# bbox_to_anchor=(0.5, 1.05),
    #                     frameon=False, fontsize=6, ncol=2)
    # for lh in leg1.legend_handles:
    #     lh.set_alpha(1)

    fig.savefig(filename, bbox_inches='tight', dpi=300)
    return fig

plot_tsne(Y, selected_trues, selected_y_pred, color_dict, class_mapping, alpha=0.15, filename="tsne_plot.png")

#%%

