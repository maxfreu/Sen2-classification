#%%

import torch
from sklearn.metrics import confusion_matrix
from torchvision.models import resnet18
import torchvision.transforms as T
from treeclassifier.utils import change_resnet_classes, change_resnet_input
from datasets import InMemoryImageClassificationDataset
from plotting import plot_confusion_matrix
from treeclassifier import utils


# import matplotlib
# matplotlib.use("TkAgg")


def filepath_to_classname(path):
    return path.split('/')[-2]


laub = ["BAH", "BI", "BPA", "BU", "EKA", "ELS", "ES", "ERL", "FAH", "GPA", "GTK", "HBU", "MBI", "MEB", "REI", "ROB",
        "SAH", "SEI", "SER", "SPA", "SPE", "STK", "TEI", "VB", "VK", "WBI", "WEI", "WER", "WES", "WPA", "ZPA"]
nadel = ["DGL", "GFI", "KI", "KTA", "SFI", "SKI", "WKI", "WTA", "ZKI", "ELA", "JLA"]
all_species = laub + nadel

class_mapping = {l : "L" for l in laub} | {n : "N" for n in nadel}
# class_mapping = {
#                     "BU": "BU",

#                     "SEI": "EI",
#                     "TEI": "EI",
#                     "REI": "EI",
#                     "ZEI": "EI",
#                     "SUEI": "EI",

#                     "GFI": "FI",
#                     "OFI": "FI",
#                     "SFI": "FI",
#                     "SWFI": "FI",
#                     "EFI": "FI",
#                     "BFI": "FI",
#                     "WFI": "FI",
#                     "SOFI": "FI",

#                     "KI": "KI",
#                     "BKI": "KI",
#                     "SKI": "KI",
#                     "RKI": "KI",
#                     "ZKI": "KI",
#                     "WKI": "KI",
#                     "MKI": "KI",
#                 "SOKI":"KI",
# }

# rest = {k: "REST" for k in all_species if k not in class_mapping.keys()}

# class_mapping = class_mapping | rest

device = "cuda" if torch.cuda.is_available() else "cpu"
# input_folder = "/tmp/classification_masked_fixed_size/"
input_folder = "/data_hdd/bkg/training/classification_masked_fixed_size/"


augmentations = torch.nn.Sequential(T.RandomVerticalFlip(),
                                    T.RandomRotation(270),
                                    T.ColorJitter(),
                                    ).to(device=device)

files = utils.get_all_image_paths(input_folder)
gen = torch.Generator().manual_seed(42)
train_files, val_files = torch.utils.data.random_split(files, lengths=[0.7, 0.3], generator=gen)
val_ds = InMemoryImageClassificationDataset(val_files,
                                            classes=None,
                                            class_mapping=class_mapping,
                                            filepath_to_classname=filepath_to_classname,
                                            augmentation=lambda x: x,
                                            device=device,
                                            nprocs=10)

#%%
model = resnet18()
change_resnet_classes(model, num_classes=len(val_ds.classes))
change_resnet_input(model, 4)
model = torch.nn.Sequential(augmentations, model)
# model.load_state_dict(torch.load("/home/max/dr/bkg_classification/weights/resnet_some_rest_weighted/resnet_some_rest_weighted.pt"))
model.load_state_dict(torch.load("./weights/resnet_laub_nadel_imbalanced/resnet_laub_nadel_imbalanced.pt"))
model = model[1]
model.eval()
model.to(device)

labels = []
preds = []
with torch.no_grad():
    # for i in utils.batched(range(len(val_ds)), 64):
    for indices in utils.batched(range(len(val_ds)), 64):
    # for indices in utils.batched(np.arange(900),4):
        # img, label = val_ds[i]
        batch = [val_ds[i] for i in indices]
        imgs = torch.stack([b[0] for b in batch])
        label = [b[1] for b in batch]
        # pred = torch.nn.functional.softmax(model(img[None,...]), dim=1)
        # pred = torch.nn.functional.softmax(model(imgs), dim=1)
        pred = model(imgs)
        pred = pred.argmax(dim=1).cpu().numpy()
        preds.extend(pred)
        labels.extend(label)


cm = confusion_matrix(labels, preds, labels=range(len(val_ds.classes)))
plot_confusion_matrix(cm, classes=val_ds.classes, fmt=".0f", outfile="./confmat_unnormalized.png", fontsize=4)
plot_confusion_matrix(cm, classes=val_ds.classes, fmt=".1f", normalize="precision", outfile="./confmat_precision.png", fontsize="xx-small")
plot_confusion_matrix(cm, classes=val_ds.classes, fmt=".1f", normalize="recall", outfile="./confmat_recall.png", fontsize="xx-small")

#%%
# plot_confusion_matrix(cm, classes=val_ds.classes, fmt=".2f", fontsize=12, normalize="precision")

#%%