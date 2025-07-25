import os
import argparse
import numpy as np
import pandas as pd
import cv2
import json
from scipy.spatial.distance import pdist
from scipy.stats import skew
from keras.models import Model, model_from_json
import keras.backend as K
from fuzzywuzzy import fuzz, process 
import matplotlib.pyplot as plt
from food_volume_estimation.depth_estimation.custom_modules import *
from food_volume_estimation.depth_estimation.project import *
from food_volume_estimation.food_segmentation.food_segmentator import FoodSegmentator
from food_volume_estimation.ellipse_detection.ellipse_detector import EllipseDetector
from food_volume_estimation.point_cloud_utils import *
import requests


class DensityDatabase():
    """Density Database searcher object. Food types are expected to be
    in column 1, food densities in column 2."""
    def __init__(self, db_path):
        """Load food density database from file or Google Sheets ID.

        Inputs:
            db_path: Path to database excel file (.xlsx) or Google Sheets ID.
        """
        if os.path.exists(db_path):
            # Read density database from excel file
            self.density_database = pd.read_excel(
                db_path, sheet_name=0, usecols=[0, 1])
        else:
            # Read density database from Google Sheets URL
            sheet = 'Sheet1'
            url = 'https://docs.google.com/spreadsheets/d/{0}/gviz/tq?tqx=out:csv&sheet={1}'.format(
                db_path, sheet)
            self.density_database = pd.read_csv(url, usecols=[0, 1],
                                                header=None)
        # Remove rows with NaN values
        self.density_database.dropna(inplace=True)

    def query(self, food):
        """Search for food density in database.

        Inputs:
            food: Food type to search for.

        Returns:
            db_entry_vals: Array containing the matched food type
            and its density.
        """
        try:
            # Search for matching food in database
            match = process.extractOne(food, self.density_database.values[:,0],
                                       scorer=fuzz.partial_ratio,
                                       score_cutoff=80)
            db_entry = (
                self.density_database.loc[
                self.density_database[
                self.density_database.columns[0]] == match[0]])
            db_entry_vals = db_entry.values
            return db_entry_vals[0]
        except:
            return ['None', 1]

class VolumeEstimator():
    """Volume estimator object."""
    def __init__(self, arg_init=True):
        """Load depth model and create segmentator object.

        Inputs:
            arg_init: Flag to initialize volume estimator with 
                command-line arguments.
        """
        if not arg_init:
            # For usage in jupyter notebook 
            print('[*] VolumeEstimator not initialized.')
        else:    
            self.args = self.__parse_args()

            # Load depth estimation model
            custom_losses = Losses()
            objs = {'ProjectionLayer': ProjectionLayer,
                    'ReflectionPadding2D': ReflectionPadding2D,
                    'InverseDepthNormalization': InverseDepthNormalization,
                    'AugmentationLayer': AugmentationLayer,
                    'compute_source_loss': custom_losses.compute_source_loss}
            with open(self.args.depth_model_architecture, 'r') as read_file:
                model_architecture_json = json.load(read_file)
                self.monovideo = model_from_json(model_architecture_json,
                                                 custom_objects=objs)
            self.__set_weights_trainable(self.monovideo, False)
            self.monovideo.load_weights(self.args.depth_model_weights)
            self.model_input_shape = (
                self.monovideo.inputs[0].shape.as_list()[1:])
            depth_net = self.monovideo.get_layer('depth_net')
            self.depth_model = Model(inputs=depth_net.inputs,
                                     outputs=depth_net.outputs,
                                     name='depth_model')
            print('[*] Loaded depth estimation model.')
            # Depth model configuration
            self.min_disp = 1 / self.args.max_depth
            self.max_disp = 1 / self.args.min_depth
            self.gt_depth_scale = self.args.gt_depth_scale

            # Create segmentator object
            self.segmentator = FoodSegmentator(self.args.segmentation_weights)

            # Plate adjustment relaxation parameter
            self.relax_param = self.args.relaxation_param

            # If given initialize food density database 
            if self.args.density_db is not None:
                self.density_db = DensityDatabase(self.args.density_db)

    def __parse_args(self):
        """Parse command-line input arguments.

        Returns:
            args: The arguments object.
        """
        # Parse command line arguments
        parser = argparse.ArgumentParser(
            description='Estimate food volume in input images.')
        parser.add_argument('--input_images', type=str, nargs='+',
                            help='Paths to input images.',
                            metavar='/path/to/image1 /path/to/image2 ...',
                            required=True)
        parser.add_argument('--depth_model_architecture', type=str,
                            help=('Depth estimation model '
                                  'architecture (.json).'),
                            metavar='/path/to/architecture.json',
                            required=True)
        parser.add_argument('--depth_model_weights', type=str,
                            help='Depth estimation model weights (.h5).',
                            metavar='/path/to/weights.h5',
                            required=True)
        parser.add_argument('--segmentation_weights', type=str,
                            help='Food segmentation model weights (.h5).',
                            metavar='/path/to/weights.h5',
                            required=True)
        parser.add_argument('--fov', type=float,
                            help='Camera Field of View (in deg).',
                            metavar='<fov>',
                            default=70)
        parser.add_argument('--plate_diameter_prior', type=float,
                            help=('Expected plate diameter (in m) '
                                  + 'or 0 to ignore plate scaling'),
                            metavar='<plate_diameter_prior>',
                            default=0.0)
        parser.add_argument('--gt_depth_scale', type=float,
                            help='Ground truth depth rescaling factor.',
                            metavar='<gt_depth_scale>',
                            default=0.35)
        parser.add_argument('--min_depth', type=float,
                            help='Minimum depth value.',
                            metavar='<min_depth>',
                            default=0.01)
        parser.add_argument('--max_depth', type=float,
                            help='Maximum depth value.',
                            metavar='<max_depth>',
                            default=10)
        parser.add_argument('--relaxation_param', type=float,
                            help='Plate adjustment relaxation parameter.',
                            metavar='<relaxation_param>',
                            default=0.01)
        parser.add_argument('--plot_results', action='store_true',
                            help='Plot volume estimation results.',
                            default=False)
        parser.add_argument('--results_file', type=str,
                            help='File to save results at (.csv).',
                            metavar='/path/to/results.csv',
                            default=None)
        parser.add_argument('--plots_directory', type=str,
                            help='Directory to save plots at (.png).',
                            metavar='/path/to/plot/directory/',
                            default=None)
        parser.add_argument('--density_db', type=str,
                            help=('Path to food density database (.xlsx) ' +
                                  'or Google Sheets ID.'),
                            metavar='/path/to/plot/database.xlsx or <ID>',
                            default=None)
        parser.add_argument('--food_type', type=str,
                            help='Food type to calculate weight for.',
                            metavar='<food_type>',
                            default=None)
        args = parser.parse_args()
        

        return args
    
    
        """
        Tạo một mảng masks_array 3D từ các ảnh PNG trong thư mục.
        
        Tham số:
        - folder_path: str: Đường dẫn đến thư mục chứa ảnh PNG.
        
        Trả về:
        - masks_array: numpy.ndarray: Mảng 3D (H, W, N), trong đó N là số lượng ảnh.
        - image_names: List[str]: Danh sách tên ảnh (không có phần mở rộng).
        """
        masks = []
        threshold = 127  # Ngưỡng chuyển ảnh sang nhị phân

        # Lấy danh sách tất cả các file PNG trong thư mục
        image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]

        if not image_paths:
            raise FileNotFoundError(f"No PNG images found in the folder: {folder_path}")

        image_names = []  # Danh sách tên ảnh
        for path in image_paths:
            # Lấy tên ảnh (không có phần mở rộng)
            image_name = os.path.splitext(os.path.basename(path))[0]
            image_names.append(image_name)

            # Đọc ảnh ở chế độ grayscale
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Image not found or could not be read: {path}")
            
            # Chuyển đổi ảnh grayscale thành ảnh nhị phân
            _, binary_mask = cv2.threshold(mask, threshold, 1, cv2.THRESH_BINARY)
            masks.append(binary_mask)

        # Gộp tất cả mặt nạ thành mảng 3D
        masks_array = np.stack(masks, axis=-1)
        return masks_array, image_names
    
    # def create_masks_array(self, folder_path):

    def create_masks_array(self, folder_path, min_size=1000):
        """
        Tạo một mảng masks_array 3D từ các ảnh PNG trong thư mục,
        bỏ qua các mask có vùng trắng quá nhỏ.

        Tham số:
        - folder_path: str: Đường dẫn đến thư mục chứa ảnh PNG.
        - min_size: int: Ngưỡng diện tích tối thiểu để giữ lại vùng trắng.

        Trả về:
        - masks_array: numpy.ndarray: Mảng 3D (H, W, N), trong đó N là số lượng ảnh.
        - image_names: List[str]: Danh sách tên ảnh (không có phần mở rộng).
        """

        masks = []  # Danh sách lưu các mask hợp lệ
        image_names = []  # Danh sách tên ảnh hợp lệ
        threshold = 127  # Ngưỡng phân tách vùng trắng và nền đen

        # Lấy danh sách tất cả file PNG trong thư mục
        image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]

        if not image_paths:
            raise FileNotFoundError(f"⚠️ Không tìm thấy ảnh PNG nào trong thư mục: {folder_path}")

        first_mask_shape = None  # Kích thước ảnh chuẩn

        for path in image_paths:
            # Lấy tên ảnh
            image_name = os.path.splitext(os.path.basename(path))[0]

            # Đọc ảnh grayscale
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"⚠️ Không thể đọc ảnh: {path}, bỏ qua.")
                continue  # Bỏ qua ảnh lỗi

            # Lưu kích thước ảnh đầu tiên làm chuẩn (H, W)
            if first_mask_shape is None:
                first_mask_shape = mask.shape

            # Resize ảnh nếu kích thước không đồng nhất
            if mask.shape != first_mask_shape:
                mask = cv2.resize(mask, (first_mask_shape[1], first_mask_shape[0]), interpolation=cv2.INTER_NEAREST)

            # Chuyển ảnh sang nhị phân (0 hoặc 255)
            _, binary_mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)

            # Tìm contours để kiểm tra vùng trắng
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Tạo mask mới (ban đầu là đen)
            filtered_mask = np.zeros_like(binary_mask)
            has_large_region = False  # Biến kiểm tra xem có vùng đủ lớn không

            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= min_size:
                    cv2.drawContours(filtered_mask, [contour], -1, (255), thickness=cv2.FILLED)
                    has_large_region = True  # Đánh dấu có ít nhất một vùng hợp lệ

            # Nếu ảnh có vùng đủ lớn, mới thêm vào danh sách
            if has_large_region:
                masks.append(filtered_mask)
                image_names.append(image_name)  # Chỉ lưu tên ảnh nếu mask hợp lệ

        # Nếu không có mask nào hợp lệ, trả về None để tránh lỗi stack
        if not masks:
            return None, []

        # Chuyển danh sách masks thành mảng 3D (H, W, N)
        masks_array = np.stack(masks, axis=-1)
        return masks_array, image_names

    def convert_volume_to_mass(self, db_path, food_volumes):
        """
        Search for food density in the database and calculate mass.

        Parameters:
            db_path (str): Path to the Excel file containing the food database.
            food_volumes (list): List of dictionaries with food type and volume in ml.

        Returns:
            list: List of dictionaries containing the matched food type and its mass in grams.
        """
        try:
            # Read the food density database from the Excel file
            density_database = pd.read_excel(db_path, sheet_name=0, usecols=[1, 2], names=["food", "density"], skiprows=1)
            
            results = []
            for item in food_volumes:
                food = item['food']
                volume = float(item['volume'])
                
                # Search for the closest matching food in the database
                match = process.extractOne(food, density_database["food"],
                                        scorer=fuzz.partial_ratio, score_cutoff=80)
                
                if match:
                    density = density_database.loc[density_database["food"] == match[0], "density"].values[0]
                    mass = density * volume  # Calculate mass in grams
                    rounded_mass = round(mass)
                    results.append({'food': food, 'mass': rounded_mass})
                    # print(f'{food} - density: {density}')
                else:
                    results.append({'food': food, 'mass': volume})  # Assume density = 1 g/ml if not found
            
            return results
        except Exception as e:
            print(f"Error: {e}")
            return [{'food': item['food'], 'mass': float(item['volume'])} for item in food_volumes]

    def get_nutrition(self, food_name, weight, api_key):
        url = f"https://api.nal.usda.gov/fdc/v1/foods/search?query={food_name}&api_key={api_key}"
        response = requests.get(url).json()
        
        if 'foods' in response and len(response['foods']) > 0:
            food = response['foods'][0]  # Lấy món ăn đầu tiên trong kết quả tìm kiếm
            nutrients = {nutrient['nutrientName']: nutrient['value'] for nutrient in food['foodNutrients']}

            # Chỉ lấy các chất cần thiết
            selected_nutrients = {
                "Calories (kcal)": nutrients.get("Energy", 0),
                "Protein (g)": nutrients.get("Protein", 0),
                "Carbohydrate (g)": nutrients.get("Carbohydrate, by difference", 0),
                "Fat (g)": nutrients.get("Total lipid (fat)", 0),
            }

            # Chuyển đổi theo khối lượng nhập vào (mặc định USDA là 100g)
            factor = weight / 100  
            nutrients_scaled = {k: round(v * factor, 2) for k, v in selected_nutrients.items()}
            
            return {food_name: nutrients_scaled}
        else:
            return {food_name: "Không tìm thấy dữ liệu!"}
    
    def get_nutrition_for_all(self, food_masses, api_key):
        results = []
        for item in food_masses:
            food_name = item['food']
            weight = item['mass']
            results.append(self.get_nutrition(food_name, weight, api_key))
        return results

    def estimate_volume(self, input_image, fov=70,  plate_diameter_prior=0.3,
            plot_results=False, para_folder_path=None, plots_directory=None):
        """Volume estimation procedure.

        Inputs:
            input_image: Path to     input image or image array.
            fov: Camera Field of View.
            plate_diameter_prior: Expected plate diameter.
            plot_results: Result plotting flag.
            plots_directory: Directory to save plots at or None.
        Returns:
            estimated_volume: Estimated volume.
        """
        # Load input image and resize to model input size
        if isinstance(input_image, str):
            img = cv2.imread(input_image, cv2.IMREAD_COLOR)
        else:
            img = input_image
            
            
        input_image_shape = img.shape
        img = cv2.resize(img, (self.model_input_shape[1],
                               self.model_input_shape[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
        
        # print("self.model_input_shape[1]: ", self.model_input_shape[1])
        # print("self.model_input_shape[0]): ", self.model_input_shape[0])

        # Create intrinsics matrix
        intrinsics_mat = self.__create_intrinsics_matrix(input_image_shape,
                                                         fov)
        intrinsics_inv = np.linalg.inv(intrinsics_mat)

        # Predict depth
        img_batch = np.reshape(img, (1,) + img.shape)
        inverse_depth = self.depth_model.predict(img_batch)[0][0,:,:,0] 
        disparity_map = (self.min_disp + (self.max_disp - self.min_disp) 
                         * inverse_depth)
        depth = 1 / disparity_map
        
        # Convert depth map to point cloud
        depth_tensor = K.variable(np.expand_dims(depth, 0))
        intrinsics_inv_tensor = K.variable(np.expand_dims(intrinsics_inv, 0))
        point_cloud = K.eval(get_cloud(depth_tensor, intrinsics_inv_tensor))
        point_cloud_flat = np.reshape(
            point_cloud, (point_cloud.shape[1] * point_cloud.shape[2], 3))

        # Find ellipse parameterss (cx, cy, a, b, theta) that 
        # describe the plate contour
        ellipse_scale = 2
        ellipse_detector = EllipseDetector(
            (ellipse_scale * self.model_input_shape[0],
             ellipse_scale * self.model_input_shape[1]))
        ellipse_params = ellipse_detector.detect(input_image)
        ellipse_params_scaled = tuple(
            [x / ellipse_scale for x in ellipse_params[:-1]]
            + [ellipse_params[-1]])

        # Scale depth map
        if (any(x != 0 for x in ellipse_params_scaled) and
                plate_diameter_prior != 0):
            print('[*] Ellipse parameters:', ellipse_params_scaled)
            # Find the scaling factor to match prior 
            # and measured plate diameters
            plate_point_1 = [int(ellipse_params_scaled[2] 
                             * np.sin(ellipse_params_scaled[4]) 
                             + ellipse_params_scaled[1]), 
                             int(ellipse_params_scaled[2] 
                             * np.cos(ellipse_params_scaled[4]) 
                             + ellipse_params_scaled[0])]
            plate_point_2 = [int(-ellipse_params_scaled[2] 
                             * np.sin(ellipse_params_scaled[4]) 
                             + ellipse_params_scaled[1]),
                             int(-ellipse_params_scaled[2] 
                             * np.cos(ellipse_params_scaled[4]) 
                             + ellipse_params_scaled[0])]
            plate_point_1_3d = point_cloud[0, plate_point_1[0], 
                                           plate_point_1[1], :]
            plate_point_2_3d = point_cloud[0, plate_point_2[0], 
                                           plate_point_2[1], :]
            plate_diameter = np.linalg.norm(plate_point_1_3d 
                                            - plate_point_2_3d)
            scaling = plate_diameter_prior / plate_diameter
        else:
            # Use the median ground truth depth scaling when not using
            # the plate contour
            print('[*] No ellipse found. Scaling with expected median depth.')
            with open(r"ourModel.txt", "a", encoding="utf-8") as f:
                    f.write('No ellipse found. Scaling with expected median depth.\n')
            predicted_median_depth = np.median(1 / disparity_map)
            scaling = self.gt_depth_scale / predicted_median_depth
        depth = scaling * depth
        point_cloud = scaling * point_cloud
        point_cloud_flat = scaling * point_cloud_flat

        # Predict segmentation masks ⭕
        folder_path = para_folder_path
        masks_array, image_names = self.create_masks_array(folder_path)

        num_masks = masks_array.shape[-1]  # Number of masks (N)
        max_cols = 4  # Maximum columns per row
        num_rows = (num_masks + max_cols - 1) // max_cols  # Calculate needed rows

        # plt.figure(figsize=(max_cols * 3, num_rows * 3))  # Dynamic figure size

        # Loop through masks and display them
        # for i in range(num_masks):
        #     mask = masks_array[:, :, i]  # Extract the ith mask
        #     food_name = image_names[i]  # Get the corresponding food name

        #     row = i // max_cols  # Compute row index
        #     col = i % max_cols  # Compute column index

        #     # plt.subplot(num_rows, max_cols, i + 1)  # Create subplot
        #     plt.title(f"{food_name}", fontsize=10)  # Add title with food name
        #     # plt.imshow(mask, cmap='gray')  # Display mask in grayscale
        #     plt.axis("off")  # Remove axes for better visualization

        # plt.tight_layout()  # Adjust layout to prevent overlap
        # # plt.show()  # Display the plot

        print(f"[*] Found {num_masks} food object(s) in the image.")        

        # Iterate over all predicted masks and estimate volumes
        estimated_volumes = []
        food_volumes = []
        
        for k in range(masks_array.shape[-1]):
            print(image_names[k])
            food_name = image_names[k]  # Lấy tên món tương ứng
             
            # Apply mask to create object image and depth map
            object_mask = cv2.resize(masks_array[:,:,k], 
                                     (self.model_input_shape[1],
                                      self.model_input_shape[0]))
            object_img = (np.tile(np.expand_dims(object_mask, axis=-1),
                                     (1,1,3)) * img)
            object_depth = object_mask * depth
            # Get object/non-object points by filtering zero/non-zero 
            # depth pixels
            object_mask = (np.reshape(
                object_depth, (object_depth.shape[0] * object_depth.shape[1]))
                > 0)
            object_points = point_cloud_flat[object_mask, :]
            non_object_points = point_cloud_flat[
                np.logical_not(object_mask), :]

            # Filter outlier points
            object_points_filtered, sor_mask = sor_filter(
                object_points, 2, 0.7)
            # Estimate base plane parameters
            plane_params = pca_plane_estimation(object_points_filtered)
            # Transform object to match z-axis with plane normal
            translation, rotation_matrix = align_plane_with_axis(
                plane_params, np.array([0, 0, 1]))
            object_points_transformed = np.dot(
                object_points_filtered + translation, rotation_matrix.T)

            # Adjust object on base plane
            height_sorted_indices = np.argsort(object_points_transformed[:,2])
            adjustment_index = height_sorted_indices[
                int(object_points_transformed.shape[0] * self.relax_param)]
            object_points_transformed[:,2] += np.abs(
                object_points_transformed[adjustment_index, 2])
             
            if (plot_results or plots_directory is not None):
                # Create object points from estimated plane
                plane_z = np.apply_along_axis(
                    lambda x: ((plane_params[0] + plane_params[1] * x[0]
                    + plane_params[2] * x[1]) * (-1) / plane_params[3]),
                    axis=1, arr=point_cloud_flat[:,:2])
                plane_points = np.concatenate(
                    (point_cloud_flat[:,:2], 
                    np.expand_dims(plane_z, axis=-1)), axis=-1)
                plane_points_transformed = np.dot(plane_points + translation, 
                                                  rotation_matrix.T)
                print('[*] Estimated plane parameters (w0,w1,w2,w3):',
                      plane_params)

                # Get the color values for the different sets of points
                colors_flat = (
                    np.reshape(img, (self.model_input_shape[0] 
                                     * self.model_input_shape[1], 3))
                    * 255)
                object_colors = colors_flat[object_mask, :]
                non_object_colors= colors_flat[np.logical_not(object_mask), :]
                object_colors_filtered = object_colors[sor_mask, :]

                # Create dataFrames for the different sets of points
                non_object_points_df = pd.DataFrame(
                    np.concatenate((non_object_points, non_object_colors), 
                                   axis=-1),
                    columns=['x','y','z','red','green','blue'])
                object_points_df = pd.DataFrame(
                    np.concatenate((object_points, object_colors), 
                    axis=-1), columns=['x','y','z','red','green','blue'])
                plane_points_df = pd.DataFrame(
                    plane_points, columns=['x','y','z'])
                object_points_transformed_df = pd.DataFrame(
                    np.concatenate((object_points_transformed, 
                                    object_colors_filtered), axis=-1),
                    columns=['x','y','z','red','green','blue'])
                plane_points_transformed_df = pd.DataFrame(
                    plane_points_transformed, columns=['x','y','z'])

                # Outline the detected plate ellipse and major axis vertices 
                plate_contour = np.copy(img)
                if (any(x != 0 for x in ellipse_params_scaled) and
                    plate_diameter_prior != 0):
                    ellipse_color = (68 / 255, 1 / 255, 84 / 255)
                    vertex_color = (253 / 255, 231 / 255, 37 / 255)
                    cv2.ellipse(plate_contour,
                                (int(ellipse_params_scaled[0]),
                                 int(ellipse_params_scaled[1])), 
                                (int(ellipse_params_scaled[2]),
                                 int(ellipse_params_scaled[3])),
                                ellipse_params_scaled[4] * 180 / np.pi, 
                                0, 360, ellipse_color, 2)
                    cv2.circle(plate_contour,
                               (int(plate_point_1[1]), int(plate_point_1[0])),
                               2, vertex_color, -1)
                    cv2.circle(plate_contour,
                               (int(plate_point_2[1]), int(plate_point_2[0])),
                               2, vertex_color, -1)

                # Estimate volume for points above the plane
                volume_points = object_points_transformed[
                    object_points_transformed[:,2] > 0]
                estimated_volume, simplices = pc_to_volume(volume_points)
                
                food_volumes.append({'food': food_name, 'volume': estimated_volume * 1000000})
                print(f'[*] Estimated volume {food_name}: {estimated_volume * 1000} L')
                with open(r"ourModel.txt", "a", encoding="utf-8") as f:
                    f.write(f'[*] Estimated volume: {estimated_volume * 1000000} mL\n')

                # Create figure of input image and predicted 
                # plate contour, segmentation mask and depth map
                os.makedirs('plots', exist_ok=True)
                images = [img, plate_contour, depth, object_img]
                image_labels = [f'Input Image {food_name}', f'Plate Contour {food_name}', f'Depth {food_name}', f'{food_name}'] 

                # Save each image with its corresponding label as a file
                # for i, (image, label) in enumerate(zip(images, image_labels)):
                #     # Define the file path
                #     file_path = os.path.join('plots', f'{label}.png')
                    
                    # Save the color image directly without plotting
                    # plt.imsave(file_path, image)  # No need for cmap for color images                


                # pretty_plotting([img, plate_contour, depth, object_img], 
                #                 (2,2),
                #                 ['Input Image', 'Plate Contour', 'Depth', 
                #                  f'{food_name}'], f"Estimated Volume {food_name}: {estimated_volume * 1000000.0:.3f} ml")
                
        
                # plt.savefig(r"E:\output_image.png")

 
                # Plot and save figure
                # if plot_results:
                #     plt.show()
                # if plots_directory is not None:
                #     if not os.path.exists(plots_directory):
                #         os.makedirs(plots_directory)
                #     (img_name, ext) = os.path.splitext(
                #         os.path.basename(input_image))
                #     filename = '{}_{}{}'.format(img_name, plt.gcf().number,
                #                                 ext)
                #     plt.savefig(os.path.join(plots_directory, filename))

                estimated_volumes.append(
                    (estimated_volume, object_points_df, non_object_points_df,
                     plane_points_df, object_points_transformed_df,
                     plane_points_transformed_df, simplices))
            else:
                # Estimate volume for points above the plane
                volume_points = object_points_transformed[
                    object_points_transformed[:,2] > 0]
                estimated_volume, _ = pc_to_volume(volume_points)
                estimated_volumes.append(estimated_volume)

        return estimated_volumes, food_volumes

    def __create_intrinsics_matrix(self, input_image_shape, fov):
        """Create intrinsics matrix from given camera fov.

        Inputs:
            input_image_shape: Original input image shape.
            fov: Camera Field of View (in deg).
        Returns:
            intrinsics_mat: Intrinsics matrix [3x3].
        """
        F = input_image_shape[1] / (2 * np.tan((fov / 2) * np.pi / 180))
        print('[*] Creating intrinsics matrix from given FOV:', fov)

        # Create intrinsics matrix
        x_scaling = int(self.model_input_shape[1]) / input_image_shape[1] 
        y_scaling = int(self.model_input_shape[0]) / input_image_shape[0] 
        intrinsics_mat = np.array(
            [[F * x_scaling, 0, (input_image_shape[1] / 2) * x_scaling], 
             [0, F * y_scaling, (input_image_shape[0] / 2) * y_scaling],
             [0, 0, 1]])
        return intrinsics_mat

    def __set_weights_trainable(self, model, trainable):
        """Sets model weights to trainable/non-trainable.

        Inputs:
            model: Model to set weights.
            trainable: Trainability flag.
        """
        for layer in model.layers:
            layer.trainable = trainable
            if isinstance(layer, Model):
                self.__set_weights_trainable(layer, trainable)

if __name__ == '__main__':
    estimator = VolumeEstimator()

    # Iterate over input images to estimate volumes
    results = {'image_path': [], 'volumes': []}
    for input_image in estimator.args.input_images:
        print('[*] Input:', input_image)
        volumes = estimator.estimate_volume(
            input_image, estimator.args.fov, 
            estimator.args.plate_diameter_prior, estimator.args.plot_results,
            estimator.args.plots_directory)

        # Store results per input image
        results['image_path'].append(input_image)
        if (estimator.args.plot_results 
            or estimator.args.plots_directory is not None):
            results['volumes'].append([x[0] * 1000 for x in volumes])
            plt.close('all')
        else:
            results['volumes'].append(volumes * 1000)

        # Print weight if density database is given
        if estimator.args.density_db is not None:
            db_entry = estimator.density_db.query(
                estimator.args.food_type)
            density = db_entry[1]
            print('[*] Density database match:', db_entry)
            # All foods found in the input image are considered to be
            # of the same type
            for v in results['volumes'][-1]:
                print('[*] Food weight:', 1000 * v * density, 'g')

    if estimator.args.results_file is not None:
        # Save results in CSV format
        volumes_df = pd.DataFrame(data=results)
        volumes_df.to_csv(estimator.args.results_file, index=False)


