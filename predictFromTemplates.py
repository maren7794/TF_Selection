import argparse
import numpy as np
import scipy.io as sio

from cmep import predict_iq_cmep
from cpm import CPM
from templateMatching import create_templates
from utils import fisher_z_transform, control_connectivities, calc_all_metrics, load_orig_data, load_raw_data
from utils_data import load_asd_data, load_hcp_data

import warnings
warnings.filterwarnings('ignore')

NETWORKS_YEO114 = {
    "VIS": [(0, 4), (56, 60)],
    "SMN": [(5, 9), (61, 65)],
    "DAN": [(10, 16), (66, 72)],
    "VAN": [(17, 26), (73, 85)],
    "LIM": [(27, 28), (86, 87)],
    "CON": [(29, 42), (88, 99)],
    "DMN": [(43, 55), (100, 112)]
}

def create_fc_from_template(raw, template_i, mi, nb_tfs, reconst_method):
    nb_samples, nb_nodes, _ = raw.shape
    template_FC = np.zeros((nb_samples, nb_nodes, nb_nodes))
    for sample_i in range(nb_samples):
        raw_sample = raw[sample_i]
        sample_mi = mi[template_i, sample_i]  # shape: nb tfs of whole ts
        selected_tfs = np.argsort(sample_mi)[-nb_tfs:]  # highest mi
        # print(raw_sample.shape, selected_tfs.shape)
        if reconst_method == 'corr':
            # print(raw_sample[:, selected_tfs].shape)
            template_FC[sample_i] = np.corrcoef(raw_sample[:, selected_tfs])
        elif reconst_method == 'sum':
            for selected_tf in selected_tfs:
                template_FC[sample_i] += np.outer(raw_sample[:, selected_tf], raw_sample[:, selected_tf])
    return template_FC


def predict_from_template(templates, raw, iq, mi, nb_tfs, control=None, reconst_method='corr', model='cmep', famID=None):
    results = []
    for t, template in enumerate(templates):
        print(t, end=': ')
        template_FC = create_fc_from_template(raw, t, mi, nb_tfs, reconst_method)
        if reconst_method == 'corr':
            template_FC = fisher_z_transform(template_FC)
        if type(control) is not type(None):
            template_FC = control_connectivities(template_FC, control)
        if model == 'cmep':
            pred, orig, _ = predict_iq_cmep(template_FC, iq, None, fam_struct=famID)
        elif model == 'cpm':
            pred, orig, _ = CPM(template_FC, iq, None, fam_struct=famID)
        results.append(calc_all_metrics(pred, orig, print_metrics=True))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--inp_path', '-i', type=str, help='Path from where to load data')
    parser.add_argument('--control', '-c', type=int, default=0, help='Control for meanFD, 0 no, 1 yes')
    parser.add_argument('--recont_type', '-r', type=str, default='corr', help='Reconstruct FC with correlation or sum.')
    parser.add_argument('--bipartition', '-b', type=int, default=1, help='Use Bipart for MI calculation?')
    parser.add_argument('--data_set', '-d', type=str, default='NKI', help='NKI, HCP or ASD')
    parser.add_argument('--asd_split', type=str, default='TD', help='TD, ASD or All')


    args = parser.parse_args()
    print(args)

    subset = ''
    if args.data_set == 'ASD':
        subset = '_' + args.asd_split

    bipart = ''
    if args.bipartition:
        bipart = 'bipart_'

    cont = ''
    if args.control:
        cont = '_cont'

    recont_type = '_' + args.recont_type

    save_name = f'./{args.data_set}{subset}_Prediction_Results_{bipart}templates_recont{recont_type}{cont}.mat'


    if args.data_set == 'NKI':
        _, _, iq, controls = load_orig_data(args.inp_path, load_dyn=False)
        raw = load_raw_data(args.inp_path)  ## is in same order as sFC & fs_iq
        control = controls[:, -1] if args.control else None
        mi = sio.loadmat('./results/Mutual_infos_NKI_Bipartitioned_0_884.mat')['mutual_infos']
        famID = None

    elif args.data_set == 'ASD':
        raw, behav_data = load_asd_data('../data/ASD')
        iq = np.array(behav_data['FSIQ'])
        mi = sio.loadmat('./results/Mutual_infos_ASD_Bipartitioned_0_200.mat')['mutual_infos']
        control, famID = None, None
        if args.control:
            raise ValueError('For ASD there is currently no control option.')
        if args.asd_split == 'TD':
            selected_participants = np.where(behav_data['Group (1=control, 2=ASD)'] == 1)
            raw, iq, mi = raw[selected_participants], iq[selected_participants], mi[:, selected_participants]
        elif args.asd_split == 'ASD':
            selected_participants = np.where(behav_data['Group (1=control, 2=ASD)'] == 2)
            raw, iq, mi = raw[selected_participants], iq[selected_participants], mi[:, selected_participants]
        mi = mi.squeeze()

    elif args.data_set == 'HCP':
        raw, control, data, iq = load_hcp_data('../data/HCP')
        if not args.control:
            control = None
        famID = data['Family_ID']
        mi = sio.loadmat('./results/Mutual_infos_HCP_Bipartitioned_0_2200.mat')['mutual_infos']
        mi = mi.squeeze()

    else:
        raise ValueError(f'Data set can either be NKI, HCP or ASD. Not {args.data_set}')

    nb_nodes = raw.shape[1]
    templates = create_templates(nb_nodes=nb_nodes) # NETWORKS_YEO114,
    pred_results = predict_from_template(templates, raw, iq, mi, reconst_method=args.recont_type,
                                         control=control, nb_tfs=44, famID=famID)

    sio.savemat(save_name, {'Res': pred_results})

    # print(np.where(np.array(pred_results).squeeze()[:, 0] > 0.344), np.where(np.array(pred_results).squeeze()[:, 0] < 9.84))




