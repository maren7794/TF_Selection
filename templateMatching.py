import argparse
import numpy as np
import itertools
from scipy.stats import pearsonr
from scipy.spatial import distance as scipy_dist
import scipy.io as sio
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score

from utils import load_orig_data, load_raw_data
from utils_data import load_asd_data, load_hcp_data

NETWORKS_YEO114 = {
    "VIS": [(0, 4), (56, 60)],
    "SMN": [(5, 9), (61, 65)],
    "DAN": [(10, 16), (66, 72)],
    "VAN": [(17, 26), (73, 85)],
    "LIM": [(27, 28), (86, 87)],
    "CON": [(29, 42), (88, 99)],
    "DMN": [(43, 55), (100, 112)]
}

NETWORKS_S100 = {
    "VIS": [(0, 6), (50, 55)],
    "SMN": [(7, 12), (56, 63)],
    "DAN": [(13, 19), (64, 69)],
    "VAN": [(20, 26), (70, 76)],
    "LIM": [(27, 29), (77, 78)],
    "CON": [(30, 36), (79, 87)],
    "DMN": [(37, 48), (88, 96)],
    "Tmp": [(49, 49), (97, 99)]
}


def load_data(data_set, inp_path):
    if data_set == 'NKI':
        raw_data = load_raw_data(inp_path)
    elif data_set == 'ASD':
        raw_data, _ = load_asd_data(inp_path)
    elif data_set == 'HCP':
        raw_data, _, _, _ = load_hcp_data(inp_path)
    else:
        raise ValueError('data set has to be NKI, ASD or HCP')
    return raw_data


def split_data(raw):
    nb_samples, nb_nodes, nb_tfs = raw.shape
    halb_nb_tfs = nb_tfs // 2
    sFC_one, sFC_two = np.zeros((nb_samples, nb_nodes, nb_nodes)), np.zeros((nb_samples, nb_nodes, nb_nodes))
    for sample_i in range(nb_samples):
        sFC_one[sample_i] = np.corrcoef(raw[sample_i, :, :halb_nb_tfs])
        sFC_two[sample_i] = np.corrcoef(raw[sample_i, :, halb_nb_tfs:])
    return sFC_one, sFC_two


def calc_distance_similarity(sFC_one, sFC_two):
    nb_samples, nb_nodes, _ = sFC_one.shape
    similiarity, distance = np.zeros((nb_samples, nb_samples)), np.zeros((nb_samples, nb_samples))
    triu_indices = np.triu_indices(nb_nodes, k=1)
    for sample_i, sample_i_FC in enumerate(sFC_one):
        sample_i_triu = sample_i_FC[triu_indices]
        for sample_j, sample_j_FC in enumerate(sFC_two):
            sample_j_triu = sample_j_FC[triu_indices]
            r, p = pearsonr(sample_i_triu, sample_j_triu)
            similiarity[sample_i, sample_j] = r
            distance[sample_i, sample_j] = scipy_dist.cosine(sample_i_triu, sample_j_triu)
    return similiarity, distance


def calc_identifiability(sim_or_dist):
    nb_samples = len(sim_or_dist)
    trace = np.trace(sim_or_dist)
    i_self = trace / nb_samples
    i_others = (np.sum(sim_or_dist) - trace) / ((nb_samples - 1) * nb_samples)

    i_diff = (i_self - i_others) * 100
    return i_diff

def select_nws(nb_nodes):
    if nb_nodes == 113:
        return NETWORKS_YEO114
    elif nb_nodes == 100:
        return NETWORKS_S100


def create_templates(nb_nodes, space='edge'):
    networks = select_nws(nb_nodes)
    templates = []
    if space == 'edge':
        temp = np.zeros((nb_nodes, nb_nodes))
    else:
        temp = np.zeros(nb_nodes)
    for i in range(1, 4):  # nb_nws+1):
        for selected_nws in itertools.combinations(networks.keys(), i):
            print(selected_nws, end='; ')
            temp *= 0
            for nw in list(selected_nws):
                regions = networks[nw]
                for (start, end) in regions:
                    temp[start:end + 1] += 1
                    if space == 'edge':
                        temp[:, start:end + 1] += 1
            temp[np.where(temp > 1)] = 0
            templates.append(temp.copy())
    return templates


def calc_temp_sim_mats(templates, raw, start, end, bipart=False, space='edge'):
    nb_samples, nb_nodes, nb_tfs = raw.shape
    nb_frames = int((end - start) * .1)
    mutual_infos = np.zeros((len(templates), nb_samples, end - start))
    template_matched_fc = np.zeros((len(templates), nb_samples, nb_nodes, nb_nodes))
    triu_indices = np.triu_indices(nb_nodes, k=1)
    for sample_i, sample_raw in enumerate(raw):
        # if sample_i % 20 == 0:
        print(sample_i, end='; ', flush=True)
        if space == 'edge':
            eFC = np.zeros((nb_nodes, nb_nodes, end - start))
            eFC_long = np.zeros((nb_nodes * (nb_nodes - 1) // 2, end - start))
            if sample_i == 0:
                print(eFC_long.shape)

        for t, template in enumerate(templates):
            print(t, end='; ', flush=True)
            if bipart:
                mi = []
                for f, frame in enumerate(range(start, end)):
                    if space == 'edge':
                        sample_frame = np.outer(raw[sample_i][:, frame], raw[sample_i][:, frame])
                        sample_frame = sample_frame[triu_indices]
                        temp = template[triu_indices].astype(int) + 1
                    else:
                        sample_frame = raw[sample_i][:, frame]
                        temp = template.astype(int) + 1
                    sample_frame = np.where(sample_frame > 0, 2, 1)
                    mut_info = mutual_info_score(sample_frame, temp)
                    mi.append(mut_info)
                mi = np.array(mi)
            else:
                mi = mutual_info_classif(eFC_long, template[triu_indices])
            mutual_infos[t, sample_i] = mi
            highest_mi = np.argsort(mi)[-nb_frames:]  # nb_frames with highest mutual information frames
            template_matched_fc[t, sample_i] = np.sum(eFC[..., highest_mi], axis=-1)
        print()
    return mutual_infos, template_matched_fc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--inp_path', '-i', type=str, help='Path from where to load data')
    parser.add_argument('--start', '-s', type=int, help='Start index of TS split')
    parser.add_argument('--end', '-e', type=int, help='End index of TS split')
    parser.add_argument('--data_set', '-d', type=str, help='Data set to use for analysis. NKI, ASD, HCP (to come)')
    parser.add_argument('--bipartition', '-bi', type=int, help='Indicates whether the eFC should be bipartitioned, 0 False, 1 True')
    parser.add_argument('--space', type=str, help='Edge or Activity space')

    args = parser.parse_args()
    print(args)
    if args.bipartition:
        bipart = 'Bipartitioned_'
    else:
        bipart = ''
    # static_con, _, fs_iq, controls = load_orig_data(args.inp_path, load_dyn=False)
    raw = load_data(args.data_set, args.inp_path)
    templates = create_templates(nb_nodes=raw.shape[1], space=args.space)
    mutual_infos, _ = calc_temp_sim_mats(templates, raw, args.start, args.end, bipart=args.bipartition)

    sio.savemat(f'./results/Mutual_infos_{args.data_set}_{bipart}{args.start}_{args.end}.mat', {'mutual_infos': mutual_infos})


