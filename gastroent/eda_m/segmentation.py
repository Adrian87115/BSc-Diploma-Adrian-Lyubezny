from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from utils import ALLOWED_EXTENSIONS

def get_paths(folder_path: str) -> list[Path]:
    """
    Obtains all images of the allowed format in the given folder.
    Allowed extensions are defined in gastroent/utils.py.

    Args:
        folder_path (str): Path to the directory to search.

    Returns:
        list[Path]: All found image file paths.
    """

    paths = []
    search_path = Path(folder_path)

    if search_path.exists():
        for file in search_path.rglob('*'):
            if file.suffix.lower() in ALLOWED_EXTENSIONS:
                paths.append(file)

    return paths

def polyp_size_distribution(paths: list[Path]) -> list[float]:
    """
    Calculates the ratio of polyp pixels to total image pixels.

    Args:
        paths (list[Path]): List of file paths to the mask images.

    Returns:
        list[float]: The fraction of the image covered by the polyp 
            (values between 0.0 and 1.0) for each mask.
    """

    stats = []

    for path in paths:
        image = Image.open(path).convert('L')
        img_array = np.array(image)

        white_occur = np.sum(img_array == 255)
        total_pixels = img_array.size 
        stats.append(white_occur / total_pixels)

    return stats

def segmentation_eda(path: str) -> None:
    """
    Performs Exploratory Data Analysis (EDA) on polyp masks.

    Calculates the size distribution of polyps, plots a histogram of the 
    percentage of image area covered, and prints summary statistics.

    Args:
        path (str): Path to the folder containing the mask images.
    """
    
    paths = get_paths(path)

    if not paths:
        print(f'No mask images found in: {path}')
        return
    
    sizes = polyp_size_distribution(paths)
    percent = np.array(sizes) * 100

    plt.figure(figsize = (8, 5))
    plt.hist(percent, bins = 100, color = 'skyblue', edgecolor = 'black')
    plt.title('Polyp Size Distribution')
    plt.xlabel('Ratio Mask (%)')
    plt.ylabel('Samples')
    plt.grid(axis = 'y', alpha = 0.75)
    plt.show()

    mean = np.mean(percent)
    median = np.median(percent)
    var = np.var(percent)
    std = np.std(percent)
    minimum = np.min(percent)
    maximum = np.max(percent)

    print(f'{path}\n'
          f'mean: {mean:.2f}%, median: {median:.2f}%, variance: {var:.2f}, std: {std:.2f}, min: {minimum:.2f}%, max: {maximum:.2f}%\n')