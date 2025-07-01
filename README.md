# Nutrition Estimation via Segmentation and Depth

## Method Overview

![Method Overview](https://res.cloudinary.com/dapvvdxw7/image/upload/v1751361408/Method_gvgahz.png)  

Our approach integrates a **U-Net architecture with a ResNet18 encoder** for depth prediction and employs **FoodSAM** for precise food segmentation. These components work together to calculate food volume and mass, which are then used to estimate nutritional content based on the USDA database. This method demonstrates robustness in handling diverse food types and complex meal compositions, offering a practical tool for dietary tracking and public health applications.  

For a detailed explanation, refer to our published paper: [Estimating Nutritional Composition from Food Volume Via Deep Learning-Based Depth and Segmentation Models](https://doi.org/10.9734/ajrcos/2025/v18i5650).  

## Introduction

Nutrition is fundamental to human health, playing a pivotal role in preventing non-communicable diseases, boosting immune function, and enhancing overall quality of life. However, dietary imbalances are a growing global concern, contributing to health crises such as obesity and malnutrition, with profound economic and health consequences. This project, **Nutrition Estimation via Segmentation and Depth**, addresses these challenges by introducing a method to estimate the nutritional content of food from a single 2D image.  

The research behind this project is published in:  

> Le, Anh, Anh Do, Thanh Nguyen, Binh Nguyen, An Tran, and Nha Tran. 2025. “Estimating Nutritional Composition from Food Volume Via Deep Learning-Based Depth and Segmentation Models”. *Asian Journal of Research in Computer Science* 18 (5):219-33. https://doi.org/10.9734/ajrcos/2025/v18i5650.  

## Key Features

- **Depth Estimation**: Predicts depth maps from single images using a self-supervised U-Net model.  
- **Food Segmentation**: Identifies up to 90 food types with high precision via FoodSAM.  
- **Volume & Mass Calculation**: Estimates food volume and converts it to mass using a comprehensive density database.  
- **Nutritional Analysis**: Retrieves detailed nutritional data from the USDA FoodData Central API.  

## Prerequisites

- **Docker**: Version 20.10 or higher (required for containerized setup).  
- **NVIDIA GPU**: Optional, with CUDA 11.1 support for accelerated performance.  
- **Internet Access**: Needed to download dependencies and pre-trained models.  

## Project Structure

- **`ckpts/`**: Pre-trained model checkpoints.  
- **`configs/`**: Configuration files for models and experiments.  
- **`datasets/`**: Directory for input datasets.  
- **`food_volume_estimation/`**: Core volume estimation algorithms.  
- **`FoodSAM/`**: Food segmentation module.  
- **`models/`**: Model architectures and weights.  
- **`Output/`**: Generated results and outputs.  
- **`Dockerfile`**: Defines the containerized environment.  
- **`requirements.txt`**: Lists all Python dependencies.  

## Dependencies

Key libraries are specified in `requirements.txt`:  
- **PyTorch**: `torch==1.8.1+cu111`, `torchvision==0.9.1+cu111`  
- **TensorFlow/Keras**: `tensorflow==1.13.1`, `keras==2.2.4`  
- **Jupyter**: `jupyter`, `notebook`, `ipykernel`  
- **Image Processing**: `scikit-image==0.16.2`, `mmcv-full==1.3.0`  
- **Web/API**: `fastapi`, `flask==1.1.1`, `uvicorn`  

To modify dependencies, update `requirements.txt` and rebuild the Docker image.  

## Model Setup

Download and place the following pre-trained models in the specified directories:  
- **SAM Model**: `sam_vit_h_4b8939.pth`  
  - [Download](https://mega.nz/file/npIlkBCQ#J7xP9Bz_dH-0vDVk0UvX1eQ5mgRLc0tz44PtcCZEwrc) → Place in `ckpts/`.  
- **SETR_MLA Model**: `iter_80000.pth`  
  - [Download](https://mega.nz/file/OgIHmLpS#bT6e5X78zB6jxVWeUd_BYffof9WL7C5wJ5UpxEJzKM0) → Place in `ckpts/SETR_MLA/`.  
- **MonoVideo Models**:  
  - `monovideo_fine_tune_food_videos.h5` [Download](https://mega.nz/file/7sBDSTaT#jnKvMdXl-q6qQW-l4k7MxKov93_lEqKjK8Vr47Kcmec)  
  - `monovideo_fine_tune_food_videos.json` [Download](https://mega.nz/file/SgJkTbwa#tKL3RCHvxeTp7aWtbP5qxB13F8CT0KwPtVKdwNst6q4)  
  - Place both in `models/fine_tune_food_videos/`.  

**Note**: Use a Mega.nz downloader or browser extension for file access.  

## Results

Our model achieves a **Mean Relative Error (MRE)** of **11.18% to 50.35%** for single food items, with consistent performance across complex meal scenarios. It outperforms baseline methods (e.g., Graikos et al., 2020) in both single and multi-food contexts, demonstrating robustness and practical utility.  

## Installation and Setup

### 1. Clone the Repository  
```bash
git clone https://github.com/<your-username>/nutrition-estimation.git  
cd nutrition-estimation  
```  

### 2. Build the Docker Image  
A `Dockerfile` is provided to create a reproducible environment. Build the image with:  
```bash
docker build -t nutrition-estimation .  
```  

### 3. Verify the Build  
Check that the image was created successfully:  
```bash
docker images  
```  

## Usage

### 1. Launch the Docker Container  
Run the container and expose port 8888 for Jupyter Notebook access:  
```bash
docker run -p 8888:8888 nutrition-estimation  
```  
- **GPU Support**: For NVIDIA GPU acceleration, use:  
```bash
docker run --gpus all -p 8888:8888 nutrition-estimation  
```  

### 2. Access Jupyter Notebook  
- Open your browser and go to: `http://localhost:8888`.  
- Retrieve the access token from the container logs (via `docker logs <container-id>`) and enter it on the Jupyter login page.  

### 3. Explore the Notebooks  
- Pre-installed Jupyter kernels and dependencies are ready for use.  
- Open and run notebooks like `api_edit_mass.ipynb` or `api_mass.ipynb` to interact with the system.  

## Troubleshooting

- **Jupyter Token Missing**: Check container logs with `docker logs <container-id>`.  
- **GPU Not Working**: Verify NVIDIA drivers and Container Toolkit installation.  
- **Dependency Issues**: Update `requirements.txt` and rebuild the image.  

## Contributing

We welcome contributions! Please submit issues or pull requests via GitHub with detailed descriptions of your changes.  

## License

[MIT License](LICENSE)

## Contact

For questions or support, contact us at [anhlh.cv@gmail.com]
