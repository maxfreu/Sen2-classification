#%%
import os
import torch

from sklearn.metrics import confusion_matrix
from plotting import plot_confusion_matrix
from omegaconf import OmegaConf
from sen2classification.datamodules import ClassificationDataModule, MultiModalClassificationDataModule
from sen2classification.classifiers import TreeClassifier, MultiModalTreeClassifier
from torchmetrics.functional import accuracy, precision, recall


#%%
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
    dm = MultiModalClassificationDataModule(**cfg.data)
    dm.setup("")
    dm.training_data.augmentation = lambda x: x

    print("data loaded")

    model = MultiModalTreeClassifier.load_from_checkpoint(checkpoint, map_location=device)
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
        for x_cnn, x_transformer, labels in dl:
            x_transformer = [x.to(device) for x in x_transformer]
            pred = model(x_cnn.to(device), x_transformer)[2].argmax(dim=1).cpu().numpy()
            all_labels.extend(labels)
            all_preds.extend(pred)

    all_labels = torch.tensor(all_labels)
    all_preds  = torch.tensor(all_preds)

    outpath = f"./confmats/{name}"
    os.makedirs(outpath, exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds, labels=range(dm.num_classes))
    plot_confusion_matrix(cm, classes=dm.training_data.image_dataset.classes, fmt=".0f", outfile=os.path.join(outpath, f"confmat_unnormalized_{version}.png"), fontsize=4)
    plot_confusion_matrix(cm, classes=dm.training_data.image_dataset.classes, fmt=".1f", normalize="precision", outfile=os.path.join(outpath, f"confmat_precision_{version}.png"), fontsize="xx-small")
    plot_confusion_matrix(cm, classes=dm.training_data.image_dataset.classes, fmt=".1f", normalize="recall", outfile=os.path.join(outpath, f"confmat_recall_{version}.png"), fontsize="xx-small")

    task = "binary" if dm.num_classes == 2 else "multiclass"
    acc = accuracy(all_preds, all_labels,  task=task, num_classes=dm.num_classes)
    pre = precision(all_preds, all_labels, task=task, num_classes=dm.num_classes)
    rec = recall(all_preds, all_labels,    task=task, num_classes=dm.num_classes)

    with open(os.path.join(outpath, f"report_{version}.txt"), "w") as f:
        f.write(str(dm.training_data.image_dataset.classes)+"\n")
        f.write(f"acc: {acc*100:.1f}\n")
        f.write(f"pre: {pre*100:.1f}\n")
        f.write(f"rec: {rec*100:.1f}\n")


#%%
if __name__ == "__main__":
    # folder = sys.argv[1]
    folders = [#"./output/5_classes/imbalanced",
               # "./output/5_classes/weighted",
               # "./output/laub_nadel/weighted",
               # "./output/laub_nadel/imbalanced",
            #    "./output/all_classes/weighted",
               # "./output/all_classes/imbalanced",
            #    "./output/laub_nadel_multimodal/weighted_simple_head_bs128",
            #    "./output/all_classes_multimodal/version_0",
               "./output/all_classes_multimodal/version_1",
               ]
    for folder in folders:
        analyze(folder, ckpt=None)

#%%


