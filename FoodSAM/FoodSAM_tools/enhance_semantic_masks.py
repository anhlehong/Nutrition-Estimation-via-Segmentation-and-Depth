import cv2
import numpy as np
import os
import logging

def calculate_single_image_masks_label(mask_file, pred_mask_file, category_list, sam_mask_label_file_name, sam_mask_label_file_dir):
    """
 mask_index, category_id, category_name, category_count, mask_count
    """
    sam_mask_data = np.load(mask_file)
    pred_mask_img = cv2.imread(pred_mask_file)[:,:,-1] # red channel
    shape_size = pred_mask_img.shape[0] * pred_mask_img.shape[1]
    logger = logging.getLogger()
    folder_path = os.path.dirname(pred_mask_file)
    sam_mask_category_folder = os.path.join(folder_path, sam_mask_label_file_dir)
    os.makedirs(sam_mask_category_folder, exist_ok=True)
    mask_category_path = os.path.join(sam_mask_category_folder, sam_mask_label_file_name)
    with open(mask_category_path, 'w') as f:
        f.write("id,category_id,category_name,category_count_ratio,mask_count_ratio\n")
        for i in range(sam_mask_data.shape[0]):
            single_mask = sam_mask_data[i]
            single_mask_labels = pred_mask_img[single_mask]
            unique_values, counts = np.unique(single_mask_labels, return_counts=True, axis=0)
            max_idx = np.argmax(counts)
            single_mask_category_label = unique_values[max_idx]
            count_ratio = counts[max_idx]/counts.sum()

            logger.info(f"{folder_path}/sam_mask/{i} assign label: [ {single_mask_category_label}, {category_list[single_mask_category_label]}, {count_ratio:.2f}, {counts.sum()/shape_size:.4f} ]")
            f.write(f"{i},{single_mask_category_label},{category_list[single_mask_category_label]},{count_ratio:.2f},{counts.sum()/shape_size:.4f}\n")

    f.close()


def predict_sam_label(data_folder, category_txt,
                      masks_path_name="sam_mask/masks.npy",
                      sam_mask_label_file_name="sam_mask_label.txt",
                      pred_mask_file_name="pred_mask.png",
                      sam_mask_label_file_dir="sam_mask_label"):

    category_lists = []
    with open(category_txt, 'r') as f:
        category_lines = f.readlines()
        category_list = [' '.join(line_data.split('\t')[1:]).strip() for line_data in category_lines]
        f.close()
        category_lists.append(category_list)
    
    for test_path, category_list in zip(data_folder, category_lists):
        img_ids = os.listdir(test_path)
        for img_id in img_ids:
            mask_file_path = os.path.join(test_path, img_id, masks_path_name)
            pred_mask_file_path = os.path.join(test_path, img_id, pred_mask_file_name)
            if os.path.exists(mask_file_path) and os.path.exists(pred_mask_file_path):
                calculate_single_image_masks_label(mask_file_path, pred_mask_file_path, category_list, sam_mask_label_file_name, sam_mask_label_file_dir)


def load_categories(category_txt):
    """
    Load category names from a category file.
    
    Args:
        category_txt (str): Path to the category file.
    
    Returns:
        list: A list of category names.
    """
    with open(category_txt, 'r') as f:
        category_lines = f.readlines()
        # Extract category names from the file
        category_list = [' '.join(line_data.split('\t')[1:]).strip()
                         for line_data in category_lines]
    return category_list


# def visualization_save(mask, save_path, img_path, color_list, category_txt):
#     """
#     Visualize and save individual masks and visualization image.

#     Args:
#         mask (np.array): Mask array with labeled regions.
#         save_path (str): Path to save the visualized image.
#         img_path (str): Path to the original image.
#         color_list (list): List of colors corresponding to labels.
#         category_txt (str): Path to the category file.
#     """
#     # Load categories
#     category_list = load_categories(category_txt)

#     # Tạo thư mục lưu mask riêng lẻ
#     masks_dir = r"masks"
#     info_txt_path = r"/masks/mask_info.txt"
#     os.makedirs(masks_dir, exist_ok=True)

#     # Tìm các giá trị duy nhất (nhãn) trong mask
#     values = set(mask.flatten().tolist())
#     final_masks = []

#     # Ghi thông tin mask
#     with open(info_txt_path, 'w') as f_info:
#         f_info.write("Label, Name, Area, BoundingBox(x0, y0, w, h)\n")

#         for v in values:
#             if v == 0:
#                 continue

#             # Tạo mask nhị phân cho nhãn hiện tại
#             binary_mask = (mask == v).astype(np.uint8) * 255
#             final_masks.append((binary_mask, v))

#             # Tính toán diện tích mask
#             area = np.sum(binary_mask > 0)

#             # Tìm bounding box
#             coords = np.column_stack(np.where(binary_mask > 0))
#             x0, y0 = coords.min(axis=0)
#             x1, y1 = coords.max(axis=0)
#             w, h = x1 - x0 + 1, y1 - y0 + 1

#             # Lấy tên món từ danh mục
#             category_name = category_list[v] if v < len(
#                 category_list) else f"Unknown_{v}"

#             # Lưu thông tin vào file
#             f_info.write(
#                 f"{v}, {category_name}, {area}, ({x0}, {y0}, {w}, {h})\n")

#             # Lưu mask nhị phân vào thư mục với tên món
#             mask_file = os.path.join(masks_dir, f"{category_name}.png")
#             cv2.imwrite(mask_file, binary_mask)

#     # Nếu không có mask thì thoát
#     if len(final_masks) == 0:
#         return

#     # Tạo hình ảnh trực quan hóa
#     h, w = mask.shape
#     result = np.zeros((h, w, 3), dtype=np.uint8)

#     for binary_mask, label in final_masks:
#         result[binary_mask > 0] = color_list[label]

#     # Đọc ảnh gốc và chồng mask lên ảnh
#     image = cv2.imread(img_path)
#     vis = cv2.addWeighted(image, 0.5, result, 0.5, 0)

#     # Lưu hình ảnh trực quan hóa
#     cv2.imwrite(save_path, vis)

def visualization_save(mask, save_path, img_path, color_list, category_txt, min_area=1500):
    """
    Visualize and save individual masks and visualization image.

    Args:
        mask (np.array): Mask array with labeled regions.
        save_path (str): Path to save the visualized image.
        img_path (str): Path to the original image.
        color_list (list): List of colors corresponding to labels.
        category_txt (str): Path to the category file.
        min_area (int): Minimum area for a mask to be considered valid.
    """
    # Load categories
    category_list = load_categories(category_txt)

    # Create folder for individual masks
    
    save_dir = os.path.dirname(save_path)  # Kết quả: "/some/path"
    # Khi cần tạo thư mục "masks" bên trong thư mục gốc
    masks_dir = os.path.join(save_dir, "masks")
    info_txt_path = os.path.join(masks_dir, "mask_info.txt")
    os.makedirs(masks_dir, exist_ok=True)
    

    # Find unique labels in the mask
    values = set(mask.flatten().tolist())
    final_masks = []

    # Open file to write mask information
    with open(info_txt_path, 'w') as f_info:
        f_info.write("Label, Name, Area, BoundingBox(x0, y0, w, h)\n")

        for v in values:
            if v == 0:
                continue

            # Create binary mask for the current label
            binary_mask = (mask == v).astype(np.uint8) * 255

            # Compute mask area
            area = np.sum(binary_mask > 0)
            
            # Skip if area is too small
            if area < min_area:
                continue

            final_masks.append((binary_mask, v))

            # Find bounding box
            coords = np.column_stack(np.where(binary_mask > 0))
            x0, y0 = coords.min(axis=0)
            x1, y1 = coords.max(axis=0)
            w, h = x1 - x0 + 1, y1 - y0 + 1

            # Get category name
            category_name = category_list[v] if v < len(category_list) else f"Unknown_{v}"

            # Write mask info to file
            f_info.write(f"{v}, {category_name}, {area}, ({x0}, {y0}, {w}, {h})\n")

            # Save binary mask
            mask_file = os.path.join(masks_dir, f"{category_name}.png")
            cv2.imwrite(mask_file, binary_mask)

    # If no valid masks, exit
    if len(final_masks) == 0:
        return

    # Create visualization image
    h, w = mask.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)

    for binary_mask, label in final_masks:
        result[binary_mask > 0] = color_list[label]

    # Read original image and overlay mask
    image = cv2.imread(img_path)
    vis = cv2.addWeighted(image, 0.5, result, 0.5, 0)

    # Save visualization image
    cv2.imwrite(save_path, vis)


def enhance_masks(data_folder, category_txt, color_list_path, num_class=104, area_thr=0, ratio_thr=0.5, top_k=80,
                  masks_path_name="sam_mask/masks.npy",
                  new_mask_label_file_name="semantic_masks_category.txt",
                  pred_mask_file_name="pred_mask.png",
                  enhance_mask_name='enhance_mask.png',
                  enhance_mask_vis_name='enhance_vis.png',
                  sam_mask_label_file_dir='sam_mask_label'):
        
    predict_sam_label([data_folder], category_txt, masks_path_name, new_mask_label_file_name, pred_mask_file_name, sam_mask_label_file_dir)
    color_list = np.load(color_list_path)
    color_list[0] = [238, 239, 20]
    for img_folder in os.listdir(data_folder):
        if img_folder == 'sam_process.log':
            continue
        category_info_path = os.path.join(data_folder, img_folder, sam_mask_label_file_dir, new_mask_label_file_name)
        sam_mask_folder = os.path.join(data_folder, img_folder)
        pred_mask_path = os.path.join(data_folder, img_folder, pred_mask_file_name)
        img_path = os.path.join(data_folder, img_folder, 'input.jpg')
        save_dir = os.path.join(data_folder, img_folder)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, enhance_mask_name)
        vis_save_path = os.path.join(save_dir, enhance_mask_vis_name)

        pred_mask = cv2.imread(pred_mask_path)[:,:,2]
        f = open(category_info_path, 'r')
        category_info = f.readlines()[1:]
        category_area = np.zeros((num_class,))
        f.close()
        for info in category_info:
            label, area = int(info.split(',')[1]), float(info.split(',')[4])
            category_area[label] += area

        category_info = sorted(category_info, key=lambda x:float(x.split(',')[4]), reverse=True)
        category_info = category_info[:top_k]
        
        enhanced_mask = pred_mask
        
        sam_masks = np.load(os.path.join(sam_mask_folder, masks_path_name))
        for info in category_info:
            idx, label, count_ratio, area = info.split(',')[0], int(info.split(',')[1]), float(info.split(',')[3]), float(info.split(',')[4])
            if area < area_thr:
                continue
            if count_ratio < ratio_thr:
                continue
            sam_mask = sam_masks[int(idx)].astype(bool)
            assert (sam_mask.sum()/ (sam_mask.shape[0] * sam_mask.shape[1]) - area) < 1e-4
            enhanced_mask[sam_mask] = label
        cv2.imwrite(save_path, enhanced_mask)
        visualization_save(enhanced_mask, vis_save_path,
                           img_path, color_list, category_txt)

