import os
import numpy as np
import argparse
import random
from utils.io import boolean_string
import torch
import torchvision.transforms as transforms
from datasets.training_dataset import HomoAffTps_Dataset
from utils.pixel_wise_mapping import remap_using_flow_fields
from matplotlib import pyplot as plt
from utils.transforms import ArrayToTensor
from utils.io import writeFlow
import flow_vis
from PIL import Image
import gc
from pathlib import Path
from glob import glob
from natsort import natsorted
resume_idx = glob('.datagen_ckpt_*')
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Change Detection Dataset Generation script')
    parser.add_argument('--image_data_path', type=str, default = '../dataset',
                        help='path to directory containing the original images.')
    parser.add_argument('--csv_path', type=str, default='datasets/csv_files/homo_aff_tps_train_DPED_CityScape_ADE.csv',
                        help='path to the CSV files containing warping params')
    parser.add_argument('--save_dir', type=str, default = '../dataset/synthetic',
                        help='path directory to save the image pairs, ground-truth flows, and change masks')
    parser.add_argument('--plot', default=False, type=boolean_string,
                        help='if true, visualize the generalized samples')
    parser.add_argument('--resume_idx', type=int,
                        default=int(natsorted(glob('.datagen_ckpt_*'))[-1].split('_')[-1])+1 if len(glob('.datagen_ckpt_*'))>0 else 0,
                        help='resume from this index')
    parser.add_argument('--seed', type=int, default=1992,
                        help='Pseudo-RNG seed')

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    plot = args.plot
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    flow_dir = os.path.join(save_dir, 'flow')
    if not os.path.exists(flow_dir):
        os.makedirs(flow_dir)
    viz_dir = os.path.join(save_dir, 'viz')
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    data_types = ['image','mask']
    img_types = ['ref','query']
    change_names = ['static','new','missing','replaced','moved']
    change_dirs = {'image':{'ref':{},'query':{}},
                   'mask':{'ref':{},'query':{}},
                   'flow':{}}

    for data_type in data_types:
        data_type_dir = os.path.join(save_dir,data_type)
        if not os.path.exists(data_type_dir): os.makedirs(data_type_dir)
        for img_type in img_types:
            img_type_dir = os.path.join(data_type_dir, img_type)
            if not os.path.exists(img_type_dir): os.makedirs(img_type_dir)
            for change_name in change_names:
                change_dir = os.path.join(img_type_dir,change_name)
                if not os.path.exists(change_dir): os.makedirs(change_dir)
                change_dirs[data_type][img_type][change_name] = change_dir
    for change_name in change_names:
        change_dir = os.path.join(flow_dir, change_name)
        if not os.path.exists(change_dir): os.makedirs(change_dir)
        change_dirs['flow'][change_name] = change_dir

    # datasets
    source_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    target_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    pyramid_param = [520]

    # training dataset
    train_dataset = HomoAffTps_Dataset(image_path=args.image_data_path,
                                       csv_file=args.csv_path,
                                       transforms=source_img_transforms,
                                       transforms_target=target_img_transforms,
                                       pyramid_param=pyramid_param,
                                       get_flow=True,
                                       output_size=(520, 520),
                                       start_idx = args.resume_idx)

    for i, minibatch in enumerate(train_dataset, start=args.resume_idx):
        SAVE_SUCCESSFUL = False
        while SAVE_SUCCESSFUL == False:
            try:
                for data_type in data_types:
                    for img_type in img_types:
                        for change_name in change_names:
                            save_data = minibatch[change_name][data_type][img_type]
                            save_path = change_dirs[data_type][img_type][change_name]
                            save_filepath = os.path.join(save_path,'{}.png'.format(i))
                            im = Image.fromarray(save_data.numpy().astype('uint8'))
                            im.save(save_filepath)

                for change_name in change_names:
                    save_data = minibatch[change_name]['flow']
                    save_path = change_dirs['flow'][change_name]
                    # save flow
                    flow_gt = minibatch[change_name]['flow'].numpy() # shape is HxWx2
                    # save the flow file and the images files
                    name_flow = '{}.flo'.format(i)


                    writeFlow(flow_gt, name_flow, save_path)

                idx_mapping = {(0,0):0,(0,1):1,(1,0):2,(1,1):3}
                # save ref
                if args.plot:
                    fig, axis = plt.subplots(5, 6, figsize=(20, 20))
                    for dtype_idx, data_type in enumerate(data_types):
                        for img_idx, img_type in enumerate(img_types):
                            for change_idx,change_name in enumerate(change_names):
                                save_data = minibatch[change_name][data_type][img_type]
                                axis[change_idx][idx_mapping[(dtype_idx,img_idx)]].imshow(save_data.numpy().astype('uint8'))
                                axis[change_idx][idx_mapping[(dtype_idx,img_idx)]].set_title('{}/{}/{}'.format(data_type,img_type,change_name))
                                axis[change_idx][idx_mapping[(dtype_idx,img_idx)]].axis('off')
                    for change_idx, change_name in enumerate(change_names):
                        flow_gt = minibatch[change_name]['flow'].numpy()  # shape is HxWx2
                        axis[change_idx][4].imshow(flow_vis.flow_to_color(flow_gt))
                        axis[change_idx][4].set_title('{}'.format('flow'))
                        axis[change_idx][4].axis('off')
                        ref_img = minibatch[change_name]['image']['ref'].numpy().astype('uint8') # h,w,3
                        mask = minibatch[change_name]['mask'] if change_name == 'moved' else None
                        remapped_gt = remap_using_flow_fields(ref_img, flow_gt[:, :, 0], flow_gt[:, :, 1],
                                                              mask=mask)
                        axis[change_idx][5].imshow(remapped_gt)
                        axis[change_idx][5].set_title('{}'.format('warped ref (w.r.t. GT flow)'))
                        axis[change_idx][5].axis('off')

                    fig.savefig(os.path.join(viz_dir, 'synthetic_pair_{}'.format(i)), bbox_inches='tight')
                    plt.close(fig)

                if len(glob(".datagen_ckpt_*")) > 0:
                    for path in glob(".datagen_ckpt_*"):
                        os.remove(path)
                Path(".datagen_ckpt_{}".format(str(i))).touch()

                SAVE_SUCCESSFUL = True
                print("[{}/{}] SAVED".format(i, len(train_dataset.df)))
                gc.collect() # release unreferenced memory
            except Exception as e:
                print(e)
                print('retrying..')
                pass
