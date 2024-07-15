#%%
import os
import torch
import argparse
from sklearn.metrics import confusion_matrix
from plotting import plot_confusion_matrix
from omegaconf import OmegaConf
from sen2classification.datamodules import TimeSeriesClassificationDataModule
from sen2classification.models.classifiers import SBERTClassifier
from torchmetrics.functional import accuracy, precision, recall


#%%
def find_checkpoints_folder(root_folder):
    """
    Recursively browse through subfolders until a folder named "checkpoints" appears.

    Args:
    root_folder (str): The path to the root folder.

    Returns:
    str: The complete folder path of the folder containing the "checkpoints" folder.
    """
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]

    # Ask user to select a subfolder index
    print("Subfolders:")
    for idx, subfolder in enumerate(subfolders):
        print(f"{idx}: {subfolder}")

    selected_index = int(input("Enter the index of the subfolder: "))
    selected_subfolder = subfolders[selected_index]

    # If the selected subfolder contains "checkpoints", return the path
    if "checkpoints" in os.listdir(selected_subfolder):
        return selected_subfolder
    else:
        # Recurse into the selected subfolder
        return find_checkpoints_folder(selected_subfolder)


def analyze(folder, ckpt=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = os.path.join(folder, "checkpoints")
    cfg = OmegaConf.load(os.path.join(folder, "config.yaml"))
    cfg.data.batch_size = 128
    name = cfg.trainer.logger.init_args.name
    version = cfg.trainer.logger.init_args.version

    checkpoints = os.listdir(checkpoint_path)
    print(checkpoints)

    if ckpt is None:
        print(folder)
        print("please enter checkpoint index")
        checkpoint = os.path.join(checkpoint_path, checkpoints[int(input())])
    else:
        checkpoint = os.path.join(checkpoint_path, checkpoints[ckpt])

    print("loading data")
    # dm = ClassificationDataModule(**cfg.data)
    dm = TimeSeriesClassificationDataModule(**cfg.data)
    dm.setup("")
    dm.training_data.augmentation = lambda x: x

    print("data loaded")

    model = SBERTClassifier.load_from_checkpoint(checkpoint, map_location=device, max_embedding_size=366 if dm.return_mode else 2600)
    # model.model = torch.nn.Sequential(torch.nn.Identity(), model.model)
    # model.model.load_state_dict(torch.load("./weights/resnet_laub_nadel_imbalanced/resnet_laub_nadel_imbalanced.pt", map_location=device))
    # model.model.load_state_dict(torch.load("deleteme.pt", map_location=device))
    # model.model = model.model[1]
    # model = utils.maybe_compile(model)
    model = model.to(device)
    model.eval()

    print("model loaded")

    dl = dm.val_dataloader()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for boa, times, mask, labels in dl:
            x_transformer = (boa, times, mask)
            x_transformer = [x.to(device) for x in x_transformer]
            pred = model(x_transformer).argmax(dim=1).cpu().numpy()
            all_labels.extend(labels)
            all_preds.extend(pred)

    all_labels = torch.tensor(all_labels)
    all_preds  = torch.tensor(all_preds)

    outpath = f"./confmats/{name}"
    os.makedirs(outpath, exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds, labels=range(dm.num_classes))
    plot_confusion_matrix(cm, classes=dm.training_data.classes, fmt=".0f", outfile=os.path.join(outpath, f"confmat_unnormalized_{version}.png"), fontsize=4)
    plot_confusion_matrix(cm, classes=dm.training_data.classes, fmt=".1f", normalize="precision", outfile=os.path.join(outpath, f"confmat_precision_{version}.png"), fontsize="xx-small")
    plot_confusion_matrix(cm, classes=dm.training_data.classes, fmt=".1f", normalize="recall", outfile=os.path.join(outpath, f"confmat_recall_{version}.png"), fontsize="xx-small")

    task = "binary" if dm.num_classes == 2 else "multiclass"
    acc = accuracy(all_preds, all_labels,  task=task, num_classes=dm.num_classes)
    pre = precision(all_preds, all_labels, task=task, num_classes=dm.num_classes)
    rec = recall(all_preds, all_labels,    task=task, num_classes=dm.num_classes)

    with open(os.path.join(outpath, f"report_{version}.txt"), "w") as f:
        f.write(str(dm.training_data.classes)+"\n")
        f.write(f"acc: {acc*100:.1f}\n")
        f.write(f"pre: {pre*100:.1f}\n")
        f.write(f"rec: {rec*100:.1f}\n")


#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get a list of all subfolders within a root folder.')
    parser.add_argument('root_folder', help='Path to the root folder')
    args = parser.parse_args(".")

    checkpoints_folder = find_checkpoints_folder(args.root_folder)

    analyze(checkpoints_folder, ckpt=None)

#%%


