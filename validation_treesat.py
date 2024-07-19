import os
import datetime
import numpy as np
from sen2classification.plotting import plot_confusion_matrix


def validate_treesat(model, treesat_dir, class_mapping, outpath, version, seq_len):
    model.eval()

    treesat_species_dict = {
        'Abies_alba': 36,
        'Acer_pseudoplatanus': 140,
        'Alnus_spec.': 210,
        'Betula_spec.': 200,
        'Cleared': -1,
        'Fagus_sylvatica': 100,
        'Fraxinus_excelsior': 120,
        'Larix_decidua': 50,
        'Larix_kaempferi': 51,
        'Picea_abies': 10,
        'Pinus_nigra': 22,
        'Pinus_strobus': 25,
        'Pinus_sylvestris': 20,
        'Populus_spec.': 220,
        'Prunus_spec.': 250,
        'Pseudotsuga_menziesii': 40,
        'Quercus_petraea': 111,
        'Quercus_robur': 110,
        'Quercus_rubra': 112,
        'Tilia_spec.': 150
    }

    classes = list(sorted(set(class_mapping.values())))
    N = len(classes)
    index_to_abbrev = {i:v for (i,v) in zip(range(N), classes)}

    l = lambda s: datetime.datetime.strptime(s[-18:-8], '%Y-%m-%d').date()

    cm = np.zeros((N,N))
    not_found = 0
    for year in (2018, 2019, 2020):
        # input_folder = "/data_ssd/treesat/validation_treesat"
        test_folders = np.loadtxt(f"/home/max/dr/force_cutouts/img_ids_{year}.txt", dtype=str)
        subfolders = [os.path.join(treesat_dir, x) for x in test_folders if os.path.isdir(os.path.join(treesat_dir, x))]
        for i, subfolder in enumerate(subfolders):
            try:
                tmin = datetime.date(year, 1, 1)
                tmax = datetime.date(year+1, 1, 1)
                pred = model.predict_force_folder(subfolder, 223, seq_len=seq_len, fname2date=l, save=False, tmin=tmin, tmax=tmax)
                pred = pred[2:-2, 2:-2]
                bn = os.path.basename(subfolder)
                true_species = "_".join(bn.split("_")[:2])
                if true_species == "Cleared_0":
                    true_species = "Cleared"
                true_species_code = treesat_species_dict[true_species]
                true_abbrev = class_mapping[true_species_code]
                true_index = classes.index(true_abbrev)

                class_ = np.bincount(pred.flatten()).argmax()
                pred_abbrev = index_to_abbrev[class_]
                pred_index = classes.index(pred_abbrev)

                cm[true_index, pred_index] += 1

            except FileNotFoundError:
                print(f"Folder {subfolder} not found.")
                not_found += 1

    acc = cm.diagonal().sum() / cm.sum()

    np.savetxt(os.path.join(outpath, "confmat_treesat.txt"), cm)

    plot_confusion_matrix(cm, classes=classes, fmt=".0f",
                          outfile=os.path.join(outpath,
                                               f"confmat_treesat_unnormalized_v={version}_seq_len={seq_len}.png"),
                          fontsize=4)

    plot_confusion_matrix(cm, classes=classes, fmt=".2f", normalize="precision", title="Precision",
                          outfile=os.path.join(outpath,
                                               f"confmat_treesat_precision_v={version}_seq_len={seq_len}.png"),
                          fontsize="xx-small")

    plot_confusion_matrix(cm, classes=classes, fmt=".2f", normalize="recall", title="Recall",
                          outfile=os.path.join(outpath,
                                               f"confmat_treesat_recall_v={version}_seq_len={seq_len}.png"),
                          fontsize="xx-small")

    return acc
