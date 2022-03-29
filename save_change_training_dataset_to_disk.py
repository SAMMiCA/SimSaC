
import os
import numpy as np
import argparse
import random
from utils.io import boolean_string
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets.training_dataset import HomoAffTps_Dataset
from utils.pixel_wise_mapping import remap_using_flow_fields
from matplotlib import pyplot as plt
from tqdm import tqdm
import imageio
from utils.image_transforms import ArrayToTensor
from utils.io import writeFlow
import flow_vis
from PIL import Image
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Change Detection Dataset Generation script')
    parser.add_argument('--image_data_path', type=str, default = '../dataset',
                        help='path to directory containing the original images.')
    parser.add_argument('--csv_path', type=str, default='datasets/csv_files/homo_aff_tps_train_DPED_CityScape_ADE.csv',
                        help='path to the CSV files')
    parser.add_argument('--save_dir', type=str, default = '../dataset/synthetic',
                        help='path directory to save the image pairs and corresponding ground-truth flows')
    parser.add_argument('--plot', default=False, type=boolean_string,
                        help='plot as examples the first 4 pairs ? default is False')
    parser.add_argument('--seed', type=int, default=1981,
                        help='Pseudo-RNG seed')

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

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
    change_names = ['static','new','missing','replaced','rotated']
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
    for change_name in ('static','rotated'):
        change_dir = os.path.join(flow_dir, change_name)
        if not os.path.exists(change_dir): os.makedirs(change_dir)
        change_dirs['flow'][change_name] = change_dir
    for change_name in ('missing','new','replaced'):
        os.system('ln -s '+os.path.join(flow_dir,'static')+' '+os.path.join(flow_dir,change_name))

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
                                       output_size=(520, 520))

    for i, minibatch in enumerate(train_dataset):
        SAVE_SUCCESSFUL = False
        while SAVE_SUCCESSFUL == False:
            try:
                for data_type in data_types:
                    for img_type in img_types:
                        for change_name in change_names:
                            save_data = minibatch[data_type][img_type][change_name]
                            save_path = change_dirs[data_type][img_type][change_name]
                            save_filepath = os.path.join(save_path,'{}.png'.format(i))
                            # print('[{}/{}] saving..{}  shape {}'.format(i,len(train_dataset),save_filepath,save_data.shape))
                            im = Image.fromarray(save_data.numpy().astype('uint8'))
                            im.save(save_filepath)

                for change_name in ('static', 'rotated'):
                    save_data = minibatch['flow'][change_name]
                    save_path = change_dirs['flow'][change_name]
                    # save flow
                    flow_gt = minibatch['flow'][change_name].permute(1,2,0).numpy() # now shape is HxWx2
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
                                save_data = minibatch[data_type][img_type][change_name]
                                axis[change_idx][idx_mapping[(dtype_idx,img_idx)]].imshow(save_data)
                                axis[change_idx][idx_mapping[(dtype_idx,img_idx)]].set_title('{}/{}/{}'.format(data_type,img_type,change_name))
                                axis[change_idx][idx_mapping[(dtype_idx,img_idx)]].axis('off')
                    for change_idx, change_name in enumerate(change_names):
                        flow_gt = minibatch['flow'][change_name].permute(1, 2, 0).numpy()  # now shape is HxWx2
                        axis[change_idx][4].imshow(flow_vis.flow_to_color(flow_gt))
                        axis[change_idx][4].set_title('{}'.format('flow'))

                        ref_img = minibatch['image']['ref'][change_name].numpy().astype('uint8') # h,w,3
                        mask = minibatch['mask'] if change_name == 'rotated' else None
                        remapped_gt = remap_using_flow_fields(ref_img, flow_gt[:, :, 0], flow_gt[:, :, 1],
                                                              mask=mask)
                        axis[change_idx][5].imshow(remapped_gt)
                        axis[change_idx][5].set_title('{}'.format('Warped ref (w.r.t. GT flow)'))
                    fig.savefig(os.path.join(viz_dir, 'synthetic_pair_{}'.format(i)), bbox_inches='tight')
                    plt.close(fig)

                SAVE_SUCCESSFUL = True

            except Exception as e:
                print(e)
                print('retrying..')
                pass
