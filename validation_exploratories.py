import os
import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
import datetime


def latin_to_output_index(latin_species, class_mapping, species_dict):
    bwi_icode = species_dict[latin_species]
    assert bwi_icode in class_mapping.keys()
    abbrev_list = list(sorted(set(class_mapping.values())))
    class_abbreviation = class_mapping[bwi_icode]
    return abbrev_list.index(class_abbreviation)


def compute_class_shares(sub_df, num_classes):
    total_area = sub_df["area"].sum()
    indices = np.arange(num_classes)
    normalized_area = np.zeros(num_classes)
    normalized_area[sub_df["output_index"]] = sub_df["area"] / total_area
    return pd.DataFrame({"EP": [sub_df["EP"].iloc[0]] * num_classes, "output_index": indices, "normalized_area": normalized_area})


def make_plot(area_shares, normalized_countmap, plotnames, class_mapping, outfile):
    fig, ax = plt.subplots(nrows=128, figsize=(3, 128))
    xl = sorted(set(class_mapping.values()))

    for (i, (name, group)) in enumerate(area_shares.groupby("EP")):
        ax[i].set_title(name, fontsize=4)
        # ax[i].set_xticks([])
        ax[i].set_ylim(0, 1)
        ax[i].tick_params(axis='both', which='major', labelsize=4)
        # ground truth
        ax[i].bar(x=xl, height=group["normalized_area"], alpha=0.6)

        # prediction
        idx = plotnames.index(name)
        ax[i].bar(x=xl, height=normalized_countmap[idx], alpha=0.6)

    fig.tight_layout()
    # fig.subplots_adjust(hspace=0)
    fig.savefig(outfile)


def kl_divergence(yp, yt):
    p = yp + 1e-7
    p /= sum(yp)
    q = yt + 1e-7
    q /= sum(yt)
    return np.sum(p * np.log(p/q))


def compute_kl_divergences(area_shares, normalized_countmap, plotnames):
    divs = []
    for (i, (name, group)) in enumerate(area_shares.groupby("EP")):
        idx = plotnames.index(name)
        yt = np.array(group["normalized_area"])
        yp = np.array(normalized_countmap[idx])
        divs.append(kl_divergence(yp, yt))
    return divs


def validate_exploratories(model,
                           input_folder,
                           class_mapping,
                           # report_outfile,
                           outpath,
                           time_encoding="absolute",
                           seq_len=128,
                           qai=223,
                           mean=np.zeros(10),
                           stddev=np.ones(10) * 10000,
                           species_codes_csv="/data_hdd/bwi/baumarten.csv",
                           exploratories_file="/data_hdd/exploratories/treedata_2018.gpkg",
                           ):
    num_classes = len(class_mapping)
    subfolders = list(os.walk(input_folder))[0][1]
    plotnames = [os.path.basename(f) for f in subfolders]
    fname2date = lambda s: datetime.datetime.strptime(s[6:16], '%Y-%m-%d').date()

    preds = [model.predict_force_folder(os.path.join(input_folder, subfolder),
                                        qai,
                                        seq_len,
                                        time_encoding=time_encoding,
                                        save=False,
                                        batch_size=128,
                                        tmin=datetime.date(2018, 1, 1),
                                        tmax = datetime.date(2019, 1, 1),
                                        fname2date=fname2date,
                                        mean=mean,
                                        stddev=stddev)
             for subfolder in subfolders]

    bwi_species = pd.read_csv(species_codes_csv, encoding="ISO-8859-1", sep=";")[["ICode", "Gattung", "Art"]]
    bwi_species = bwi_species[bwi_species["ICode"] < 900]
    bwi_species["latin"] = [str(g) + " " + str(s) for i, (g,s) in bwi_species[["Gattung", "Art"]].iterrows()]
    species_dict = {sp: code for code, sp in zip(bwi_species["ICode"], bwi_species["latin"])}
    missing = {
        "Acer spec": 140,
        "Alnus spec": 210,
        "Betula spec": 200,
        "Carya ovata": 190,
        "Larix spec": 50,
        "Pyrus pyraster": 293,
        "Quercus spec": 110,
        "Salix caprea": 240,
        "Salix spec": 240,
        "Tilia cordata": 150,
        "Tilia platyphyllos": 150,
        "Tilia spec": 150,
        "Ulmus glabra": 170,
        "Ulmus spec": 170,
        }
    species_dict = species_dict | missing

    df = gpd.read_file(exploratories_file)
    df["area"] = (df["d"] / 2 / 100)**2 * 3.141592654
    df["species"] = [s.replace("_", " ") for s in df["species"]]
    df["gattung"] = [s.split(" ")[0] for s in df["species"]]
    df["icode"] = [species_dict[sp] for sp in df["species"]]
    df["output_index"] = [latin_to_output_index(ln, class_mapping, species_dict) for ln in df["species"]]

    species_areas = df.groupby(["EP", "output_index"])["area"].sum().reset_index()
    area_shares = species_areas.groupby(["EP"]).apply(lambda x: compute_class_shares(x, num_classes)).droplevel("EP")

    classes = [np.bincount(p.flatten()).argmax() for p in preds]
    normalized_countmap = [np.bincount(p.flatten()) / np.bincount(p.flatten()).sum() for p in preds]
    normalized_countmap = [np.pad(cm, (0, num_classes - len(cm))) for cm in normalized_countmap]
    major_species = area_shares.sort_values("normalized_area").groupby(["EP"]).tail(1).sort_values("EP")

    major_species["y_pred"] = np.zeros(len(major_species), dtype=int)
    # major_species["icode"] = [species_dict[sp] for sp in major_species["species"]]

    for (plotname, cls_) in zip(plotnames, classes):
        major_species["y_pred"][major_species["EP"] == plotname] = cls_

    # major_species["y_true"] = [abbrev_list.index(class_mapping[icode]) for icode in major_species["icode"]]
    major_species = major_species.rename(columns={"output_index": "y_true"})
    acc = sum(major_species["y_true"] == major_species["y_pred"]) / len(major_species)
    kl_div = np.mean(compute_kl_divergences(area_shares, normalized_countmap, plotnames))

    plotfile = os.path.join(outpath, f"distribution_plot_seq_len={seq_len}.pdf")
    make_plot(area_shares, normalized_countmap, plotnames, class_mapping, plotfile)

    return acc, kl_div
