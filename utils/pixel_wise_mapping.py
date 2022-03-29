import numpy as np
import cv2


def remap_using_flow_fields(image, disp_x, disp_y, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT,
                            mask=None):
    """
    opencv remap : carefull here mapx and mapy contains the index of the future position for each pixel
    not the displacement !
    map_x contains the index of the future horizontal position of each pixel [i,j] while map_y contains the index of the future y
    position of each pixel [i,j]

    All are numpy arrays
    :param image: image to remap, HxWxC
    :param disp_x: displacement on the horizontal direction to apply to each pixel. must be float32. HxW
    :param disp_y: isplacement in the vertical direction to apply to each pixel. must be float32. HxW
    :return:
    remapped image. HxWxC
    """
    h_scale, w_scale=image.shape[:2]

    # estimate the grid
    X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                       np.linspace(0, h_scale - 1, h_scale))
    map_x = (X+disp_x).astype(np.float32)
    map_y = (Y+disp_y).astype(np.float32)
    if mask is not None:
        ref_mask =mask['ref']['rotated']
        query_mask = mask['query']['rotated']

        obj_removed_img = (~ref_mask.bool()).numpy() * image.copy()
        remapped_image_without_obj = cv2.remap(obj_removed_img, map_x, map_y, interpolation=interpolation, borderMode=border_mode)
        obj_only_img = (ref_mask.bool()).numpy() * image.copy()
        remapped_mask = cv2.remap(ref_mask.float().numpy(),map_x,map_y,interpolation,border_mode)
        # import matplotlib.pyplot as plt
        # plt.figure(2)
        # plt.subplot(121)
        # plt.imshow(query_mask.numpy())
        # plt.subplot(122)
        # plt.imshow(remapped_mask)
        # plt.show()
        # plt.figure(1)
        remapped_image_only_obj = cv2.remap(obj_only_img, map_x, map_y, interpolation=interpolation, borderMode=border_mode)
        # remapped_image = (remapped_mask.astype('bool')) * remapped_image_only_obj \
        #              + (1-remapped_mask.astype('bool')) * remapped_image_without_obj
        remapped_image = (query_mask.bool()).numpy() * remapped_image_only_obj \
                     + (~query_mask.bool()).numpy() * remapped_image_without_obj

    else:
        remapped_image = cv2.remap(image, map_x, map_y, interpolation=interpolation, borderMode=border_mode)

    return remapped_image


def remap_using_correspondence_map(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT):
    """
    opencv remap :
    attention: mapping from destination to inverse, the map given must be the inverse
    carefull here map_x and map_y contains the index of the future position for each pixel
    not the displacement !
    mapx contains the index of the future horizontal position of each pixel [i,j] while mapy contains the index of the future y
    position of each pixel [i,j]

    All are numpy arrays
    :param image: image to remap, HxWxC
    :param map_x: horizontal index of remapped position of each pixel. must be float32. HxW
    :param map_y: vertical index of remapped position of each pixel. must be float32. HxW
    :return:
    remapped image. HxWxC
    """
    remapped_image = cv2.remap(image, map_x, map_y, interpolation=interpolation, borderMode=border_mode)
    return remapped_image

