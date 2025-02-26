import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
import experiments.train_and_validate
from sen2classification import utils
from sen2classification.datasets import InMemoryTimeSeriesDataset
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def random_color():
    return np.random.choice(range(256), size=3) / 255


#%%
data_config_path = "output/time_encoding_GRU/time_encoding=doy-return_mode=single/config.yaml"
model_config_path = data_config_path
ckpt = "/home/max/dr/Sen2-classification/output/time_encoding_GRU/time_encoding=doy-return_mode=single/checkpoints/epoch=23-step=53256-val_loss=0.7466.ckpt"


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
data = experiments.train_and_validate.load_data(data_args=dataconfig | {"where": "tree_id < -70000"})
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
    plot_ids=dataconfig["plot_ids"] if "plot_ids" in dataconfig.keys() else None,
    # where="" if "plot_ids" in dataconfig.keys() else "is_train = FALSE",
    where="is_train = FALSE and species > 0 and is_pure",
    mean=dataconfig["mean"],
    stddev=dataconfig["stddev"]
)


#%%
preds = []
trues = []
dl = DataLoader(test_dataset)

with torch.no_grad():
    for (i, batch) in enumerate(dl):
        if i%100 == 0:
            print(f"{i} / {len(dl)}")
        tree_ids, boa, doy, mask, y_true = batch
        # boa = boa.to(device)
        # doy = doy.to(device)
        # mask = mask.to(device)
        y_pred = model(boa, doy, mask)
        preds.extend(y_pred.numpy())
        trues.extend(y_true.numpy())

#%%
preds = np.array(preds)
trues = np.array(trues)
y_pred = preds.argmax(axis=1)

rand_selection = np.random.rand(len(preds)) > 0.

selected_preds = preds[rand_selection]
selected_trues = trues[rand_selection]
selected_y_pred = y_pred[rand_selection]
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
color_dict = {i: random_color() for i in range(14)}

# thanks gpt
color_dict = {
    0: '#ADD8E6',  # AL (Alder)
    1: '#7C4D4C',  # ASH (Ash)
    2: '#FF0000',  # BE (Beech)
    3: '#708090',  # BG (Background) - Slate gray, neutral or background species color
    4: '#F4A300',  # BI (Birch)
    5: '#006400',  # DG (Douglas Fir)
    6: '#228BBB',  # FIR (Fir)
    7: '#AFA700',  # LA (Larch)
    8: '#8B0000',  # MA (Maple)
    9: '#0000FF',  # OA (Oak)
    10: '#4B0082',  # OC (Other Coniferous)
    11: '#B8860B',  # OD (Other Deciduous)
    12: '#FFFF00',  # PI (Pine)
    13: '#00FF00',  # SP (Spruce)
}

class_mapping = {
    0: 'Erle',
    1: 'Esche',
    2: 'Buche',
    3: 'Hintergr.',
    4: 'Birke',
    5: 'Douglasie',
    6: 'Tanne',
    7: 'Lärche',
    8: 'Ahorn',
    9: 'Eiche',
    10: 'A. Konif.',
    11: 'A. Laubb.',
    12: 'Kiefer',
    13: 'Fichte'
}

#%%
colors = [color_dict[i] for i in trues]
# class_dict = {}

fig, ax = plt.subplots(1,2)
for i in reversed(range(14)):
# for i in range(14):
    tsne_pts = Y[selected_trues == i]
    ax[0].scatter(tsne_pts[:,0], tsne_pts[:,1], color=color_dict[i], label=class_mapping[i], alpha=0.2)
ax[0].set_aspect('equal')
leg = ax[0].legend(frameon=False)
for lh in leg.legend_handles:
    lh.set_alpha(1)
ax[0].axis("off")

false_pts = Y[selected_trues != selected_y_pred]
ax[1].scatter(false_pts[:,0], false_pts[:,1], color="red", label="Wrong", alpha=0.1)
true_pts = Y[selected_trues == selected_y_pred]
ax[1].scatter(true_pts[:,0], true_pts[:,1], color="green", label="Right", alpha=0.1)
ax[1].set_aspect('equal')
leg = ax[1].legend(frameon=False)
for lh in leg.legend_handles:
    lh.set_alpha(1)
ax[1].axis("off")

fig.tight_layout()

print("Acc:", len(true_pts) / (len(true_pts) + len(false_pts)))
#%%

