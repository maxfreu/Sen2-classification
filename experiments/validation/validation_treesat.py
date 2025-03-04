import os
import ast
import datetime
import numpy as np
from .validation_exploratories import kl_divergence


def validate_treesat(model, treesat_dir, class_mapping, time_encoding, return_mode, seq_len, qai, mean, stddev, append_ndvi):
    model = model.eval()

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

    treesat_genus_dict = {
        "Abies": 36,
        "Acer": 140,
        "Alnus": 210,
        "Betula": 200,
        "Cleared": -1,
        "Fagus": 100,
        "Fraxinus": 120,
        "Larix": 50,
        "Picea": 10,
        "Pinus": 20,
        "Populus": 220,
        "Prunus": 250,
        "Pseudotsuga": 40,
        "Quercus": 110,
        "Tilia": 150
    }

    # classes present in network output
    classes = list(sorted(set(class_mapping.values())))
    num_classes = len(classes)

    per_plot_res = []
    total_kl_div = 0

    # mapping from class index to abbreviation
    # index_to_abbrev = {i:v for (i,v) in zip(range(N), classes)}
    genus2classindex = {genus: classes.index(class_mapping[icode]) for (genus, icode) in treesat_genus_dict.items()}

    fname2date = lambda s: datetime.datetime.strptime(s[-18:-8], '%Y-%m-%d').date()

    # ok, now we compute the accuracy over all three years present in the treesat dataset
    # and sum the result up into a confusion matrix
    cm = np.zeros((num_classes,num_classes))
    not_found = 0
    for year in (2018, 2019):
        # look up test folders in text files
        test_folders = np.loadtxt(f"/home/max/dr/force_cutouts/img_ids_shares_{year}.txt", dtype=str, delimiter=';')
        folders_and_shares = [(s[0], ast.literal_eval(s[1])) for s in test_folders]
        folders_and_shares = [(os.path.join(treesat_dir, folder), share) for (folder, share) in folders_and_shares if os.path.isdir(os.path.join(treesat_dir, folder))]
        for i, (subfolder, share) in enumerate(folders_and_shares):
            try:
                # compute array of true area share per species
                share_true = np.zeros(num_classes)
                for (k,v) in share.items():
                    share_true[genus2classindex[k]] += v

                # time window for data loading (tmax is exclusive)
                if return_mode=="double":
                    tmin = datetime.date(year-1, 1, 1)
                elif return_mode=="single" or return_mode=="random":
                    tmin = datetime.date(year, 1, 1)
                else:
                    raise ValueError(f"Expected return mode 'single', 'random' or 'double' but got {return_mode}")
                tmax = datetime.date(year+1, 1, 1)

                pred = model.predict_force_folder(
                    subfolder,
                    qai,
                    seq_len=seq_len,
                    fname2date=fname2date,
                    save=False,
                    tmin_data=tmin,
                    tmax_data=tmax,
                    mean=mean,
                    stddev=stddev,
                    time_encoding=time_encoding,
                    apply_argmax=False,
                    num_classes=num_classes,
                    append_ndvi=append_ndvi
                )

                # predictions are scaled to uint8 0-255 to save memory, scale back:
                pred = pred / 255
                pred = pred[2:5, 2:5, :]
                
                # infer true species name from folder name
                bn = os.path.basename(subfolder)
                true_species = "_".join(bn.split("_")[:2])
                if true_species == "Cleared_0":
                    true_species = "Cleared"

                # map true species name to NFI species code
                true_species_code = treesat_species_dict[true_species]

                # map true species code to class index within the class mapping that maps from icode to abbrev
                true_abbrev = class_mapping[true_species_code]

                # map true class abbrev to index in the class list
                true_index = classes.index(true_abbrev)

                # get the index of the class with the highest number of pixels in the prediction
                share_pred = np.bincount(pred.argmax(axis=-1).flatten()) / np.prod(pred.shape[:2])
                share_pred = np.pad(share_pred, (0, num_classes - len(share_pred)))

                pred_index = share_pred.argmax()

                kl_div = kl_divergence(share_pred, share_true)
                total_kl_div += kl_div
                per_plot_res.append([subfolder, kl_div, share_pred])

                # actually useless
                # map class index to class abbreviation
                # pred_abbrev = index_to_abbrev[class_]
                # map class abbreviation to index in the class list
                # pred_index = classes.index(pred_abbrev)

                cm[true_index, pred_index] += 1

            except FileNotFoundError:
                print(f"Folder {subfolder} not found.")
                not_found += 1

    acc = cm.diagonal().sum() / cm.sum()

    per_plot_res.sort(key=lambda x: x[1], reverse=True)

    return acc, cm, total_kl_div / len(per_plot_res), np.array(per_plot_res, dtype=object)
