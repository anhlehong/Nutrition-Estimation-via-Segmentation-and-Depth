import sys

sys.path.append('.')
sys.path.append('./SAM')
sys.path.append('./mmseg')

import argparse
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from mmcv.utils import DictAction
import argparse
import json
import os
import numpy as np
from typing import Any, Dict, List
import shutil, logging
from FoodSAM_tools.predict_semantic_mask import semantic_predict
from FoodSAM_tools.enhance_semantic_masks import enhance_masks
from FoodSAM_tools.evaluate_foodseg103 import evaluate

parser = argparse.ArgumentParser(
    description=(
        "Runs SAM automatic mask generation and semantic segmentation on an input image or directory of images, "
        "and then enhance the semantic masks based on SAM output masks"
    )
)

parser.add_argument(
    "--img_path",
    type=str,
    default=None,
    help="dir name of imgs.",
)
parser.add_argument(
    "--output",
    type=str,
    default='Output/Semantic_Results',
    help=(
        "Path to the directory where results will be output. Output will be a folder "
    ),
)
parser.add_argument("--device", type=str, default="cpu",
                    help="The device to run generation on.")


parser.add_argument(
    "--SAM_checkpoint",
    type=str,
    default="ckpts/sam_vit_h_4b8939.pth",
    help="The path to the SAM checkpoint to use for mask generation.",
)
parser.add_argument('--semantic_config', default="configs/SETR_MLA_768x768_80k_base.py", help='test config file path of mmseg')
parser.add_argument('--semantic_checkpoint', default="ckpts/SETR_MLA/iter_80000.pth", help='checkpoint file of mmseg')
parser.add_argument(
    "--model-type",
    type=str,
    default='vit_h',
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)


parser.add_argument('--color_list_path', type=str, default="FoodSAM/FoodSAM_tools/color_list.npy", help='the color used to draw for each label')

parser.add_argument(
    "--category_txt",
    default="FoodSAM/FoodSAM_tools/category_id_files/foodseg103_category_id.txt", help='the category name of each label'
)
parser.add_argument(
    "--num_class",
    default=104, help='the total number of classes including background'
)




def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    os.makedirs(os.path.join(path, "sam_mask"), exist_ok=True)
    masks_array = []
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        masks_array.append(mask.copy())
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, "sam_mask" ,filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)

    masks_array = np.stack(masks_array, axis=0)
    np.save(os.path.join(path, "sam_mask" ,"masks.npy"), masks_array)
    metadata_path = os.path.join(path, "sam_metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))
    return

def create_logger(save_folder):
    
    log_file = f"sam_process.log"
    final_log_file = os.path.join(save_folder, log_file)

    logging.basicConfig(
        format=
        '[%(asctime)s] [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(final_log_file, mode='w'),
            logging.StreamHandler()
        ])                        
    logger = logging.getLogger()
    print(f"Create Logger success in {final_log_file}")
    return logger

def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output, exist_ok=True)
    logger = create_logger(args.output)
    logger.info("running sam!")
    
    # ----------------------------------------------------------------
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available. Switching to CPU.")
        args.device = "cpu"
    # ----------------------------------------------------------------
    

    sam = sam_model_registry[args.model_type](checkpoint=args.SAM_checkpoint)
    _ = sam.to(device=args.device)
    output_mode = "binary_mask"
    amg_kwargs = {}

    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)
    
    assert args.img_path, "Provide img_path."
    targets = [args.img_path]

    for t in targets:
        logger.info(f"Processing '{t}'...")
        image = cv2.imread(t)
        if image is None:
            logger.error(f"Could not load '{t}' as an image, skipping...")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = generator.generate(image)
        base = os.path.basename(t)
        base = os.path.splitext(base)[0]
        save_base = os.path.join(args.output, base)
        os.makedirs(save_base, exist_ok=True)
        write_masks_to_folder(masks, save_base)
        shutil.copyfile(t, os.path.join(save_base, "input.jpg"))
    logger.info("sam done!\n")

    
    logger.info("running semantic seg model!")
    semantic_predict(args.semantic_config, args.semantic_checkpoint,
                     args.output, args.color_list_path, args.img_path, device=args.device)
    logger.info("semantic predict done!\n")
    

    logger.info("enhance semantic masks")
    enhance_masks(args.output, args.category_txt, args.color_list_path, num_class=args.num_class)
    logger.info("enhance semantic masks done!\n")

    logger.info("The results saved in {}!\n".format(args.output))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
