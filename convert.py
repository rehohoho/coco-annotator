
import numpy as np
import os
from PIL import Image
import argparse

from pycocotools.coco import COCO


def imgid_to_segmap(coco, imgId, ignore_overlap=False):
    """
    Convert COCO GT or results for a single image to a segmentation map
    Checks every pixel can have at most one label

    Args:
        coco:           an instance of the COCO API (ground-truth or result)
        imgId:          the id of the COCO image
        ignore_overlap: whether to ignore more than one label
    
    Returns: 
        labelMap - [h x w] segmentation map that indicates the label of each pixel
    """
    
    curImg = coco.imgs[imgId]
    imageSize = (curImg['height'], curImg['width'])
    labelMap = np.zeros(imageSize)

    # Get annotations of the current image (may be empty)
    imgAnnots = [a for a in coco.anns.values() if a['image_id'] == imgId]
    annIds = coco.getAnnIds(imgIds=imgId, iscrowd=False)
    imgAnnots = coco.loadAnns(annIds)

    # Combine all annotations of this image in labelMap
    for a in range(0, len(imgAnnots)):
        labelMask = coco.annToMask(imgAnnots[a]) == 1
        newLabel = imgAnnots[a]['category_id']

        if not ignore_overlap and (labelMap[labelMask] != 0).any():
            raise Exception('Error: Some pixels have more than one label (image %d)!' % (imgId))

        labelMap[labelMask] = newLabel

    return labelMap


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--coco_json',
        required=True,
        help='path to json file exported from coco-annotator'
    )
    parser.add_argument(
        '-o', '--output_folder',
        required=True,
        help='path to output image masks'
    )
    parser.add_argument(
        '-io', '--ignore_overlap',
        required=True,
        action='store_true',
        help='whether to ignore overlapping labels'
    )

    args = parser.parse_args()
    coco = COCO(args.coco_json)
    ids = list( coco.imgs.keys() )

    for img_id in ids:
        path = coco.loadImgs(img_id)[0]['file_name']
        path = os.path.join(args.output_folder, path)
        mask = imgid_to_segmap(coco, img_id, args.ignore_overlap)
        
        mask_im = Image.fromarray( np.array(mask, dtype=np.uint8) )
        mask_im.save(path)

        print( 'Saved mask for %s image at %s' %(img_id, path) )