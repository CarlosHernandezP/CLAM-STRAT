import pdb
import os
import pandas as pd
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits, MultiTask_Dataset
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str, choices=['tcga_cvpr', 'strat', 'xxmm_2023', 'xxmm_slnb'])
parser.add_argument('--val_frac', type=float, default= 0.1,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.1,
                    help='fraction of labels for test (default: 0.1)')

args = parser.parse_args()

if args.task == 'tcga_cvpr':
    args.n_classes=2
    csv_dir = '~/PhD/marato-derma/derma/sol/stratification/histo_csvs/tcga_all_labels.csv'

    dataset = MultiTask_Dataset(csv_path = csv_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0 :0, 1 :1},
                            patient_strat= False,
                            label_col = 'braf_positivity',
                            ignore=[])

elif args.task == 'xxmm_slnb':
    args.n_classes=2
    df_path = '/home/carlos.hernandez/datasets/csvs/xxmm_histopathology_noimputing.csv'

    dataset = Generic_WSI_Classification_Dataset(csv_path = '/home/carlos.hernandez/datasets/csvs/xxmm_histopathology_noimputing.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'negative':0, 'positive':1},
                            patient_strat= False,
                            ignore=[])

elif args.task == 'xxmm_2023':
    args.n_classes=2
    data_dir = '~/datasets/csvs/new_data_2023/histos_good_ids.csv'

    dataset = Generic_WSI_Classification_Dataset(csv_path = '/home/carlos.hernandez/datasets/csvs/new_data_2023/histos_good_ids.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0 :0, 1 :1},
                            patient_strat= False,
                            ignore=[])

elif args.task == 'strat':
    
    data_dir = '~/datasets/csvs/new_data_2023/histos_good_ids.csv'
    dataset = Generic_WSI_Classification_Dataset(csv_path = '/home/carlos.hernandez/datasets/csvs/new_data_2023/histos_good_ids.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0 :0, 1 :1},
                            patient_strat= False,
                            ignore=[])

else:
    raise NotImplementedError

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)

if __name__ == '__main__':
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        split_dir = 'splits/'+ str(args.task) + '_{}'.format(int(lf * 100))
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))



