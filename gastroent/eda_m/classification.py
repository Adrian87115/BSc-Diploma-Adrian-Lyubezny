from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from typing import Any

from utils import ALLOWED_EXTENSIONS

def get_classes_paths(folder_path: str) -> dict[str, list[Path]]:
    """
    Traverses a directory to group image paths by their parent folder name.

    Uses a depth-first search to find all images with allowed extensions within 
    the given directory. The name of the immediate parent folder containing 
    the image is treated as its class label.

    Args:
        folder_path (str): The root directory to start the search from.

    Returns:
        dict[str, list[Path]]: A dictionary where the keys are class names (strings) 
            and the values are lists of image file paths (Path objects). Returns 
            an empty dictionary if the folder_path does not exist or is not a directory.
    """

    paths_dict = {}
    main_path = Path(folder_path)

    if not (main_path.exists() and main_path.is_dir()):
        return paths_dict

    def traverse_dfs(current_path: Path) -> None:
        """
        Helping function to obtain all necessary classes with their images.
        
        Args:
            current_path (Path): Entry path of traversing.
        """

        for item in sorted(current_path.iterdir()):
            if item.is_dir():
                traverse_dfs(item)
            elif item.is_file() and item.suffix.lower() in ALLOWED_EXTENSIONS:
                class_name = current_path.name
                
                if class_name not in paths_dict:
                    paths_dict[class_name] = []
                
                paths_dict[class_name].append(item)

    traverse_dfs(main_path)
    
    return paths_dict

def plot_heatmap(stats: dict[str, dict[str, Any]], channel: str, config: dict[str, str]) -> None:
    """
    Plots a heatmap of a specific color statistic for each class.

    Visualizes the distribution of a given color channel across all classes 
    using a 2D color mesh. The y-axis represents the classes, and the x-axis 
    represents the channel's bin edges.

    Args:
        stats (dict[str, dict[str, Any]]): Dictionary mapping class names to their 
            respective color statistics. Expected to contain 'hist' and 'edges'
            data for the given channel.
        channel (str): The color channel key to plot (e.g., 'h', 's', 'v', 'c').
        config (dict[str, str]): Configuration dictionary containing plot 
            metadata, specifically 'title' and 'xlabel' keys.
    """
    
    classes = []
    hist_matrix = []
    edges = None

    for class_name, class_stats in stats.items():
        channel_data = class_stats.get(channel, {})
        hist = channel_data.get('hist')

        if hist is not None:
            classes.append(class_name)
            hist_matrix.append(hist)

            if edges is None:
                edges = channel_data.get('edges')

    if not hist_matrix or edges is None:
        return

    hist_matrix = np.array(hist_matrix)
    fig_height = max(3, len(classes) * 0.35)

    plt.figure(figsize = (10, fig_height))

    x, y = np.meshgrid(edges, np.arange(len(classes) + 1))
    mesh = plt.pcolormesh(x, y, hist_matrix, cmap = 'viridis', edgecolors = 'none')

    plt.colorbar(mesh, label = 'Average proportion of pixels')
    plt.yticks(np.arange(len(classes)) + 0.5, classes)
    plt.gca().invert_yaxis()
    plt.title(config['title'])
    plt.xlabel(config['xlabel'])
    plt.tight_layout()
    plt.show()

def rgb_to_hsvcr(rgb: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates HSVCR channels from an RGB image.

    Args:
        rgb (Image.Image): Image in RGB format.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple of four 2D 
            float32 NumPy arrays:
            - h: Hue in range [0, 360) representing the dominant color.
            - s: Saturation in range [0, 1] representing vividness or dullness.
            - v: Value in range [0, 1] representing brightness.
            - c: Chroma in range [0, 1] representing color intensity or purity.
    """

    img = np.asarray(rgb, dtype = np.float32) / 255.0
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    cmax = np.max(img, axis = -1)
    cmin = np.min(img, axis = -1)
    
    c = cmax - cmin
    v = cmax

    s = np.zeros_like(cmax)
    mask_v = cmax > 0
    s[mask_v] = c[mask_v] / cmax[mask_v]

    h = np.zeros_like(cmax)
    mask_cr = c > 0

    # Masks for which channel is max, mutually exclusive to avoid double counting - tiebreaker
    mask_r = (cmax == r) & mask_cr
    mask_g = (cmax == g) & mask_cr & ~mask_r
    mask_b = (cmax == b) & mask_cr & ~mask_r & ~mask_g

    h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / c[mask_r])) % 360
    h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / c[mask_g]) + 120) % 360
    h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / c[mask_b]) + 240) % 360

    return h, s, v, c

def get_class_color_histograms(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """
    Calculates average Hue, Saturation, Value, and Chroma histograms for a class.

    Reads a list of image paths, computes the HSVCR channels for each, builds 
    normalized histograms, and averages them across all images. Hue calculations 
    ignore pixels with very low saturation (s <= 0.05) to avoid noisy data.

    Args:
        paths (list[Path]): A list of file paths to the images to be processed.

    Returns:
        dict[str, dict[str, Any]]: A dictionary containing the averaged histograms 
            and bin edges for each color channel ('h', 's', 'v', 'c'). Each channel 
            maps to a dictionary with keys:
            - 'hist': A 1D float64 NumPy array representing the averaged normalized 
              histogram (or None if no valid data was found).
            - 'edges': A 1D NumPy array of the bin edges used for the histogram.
    """

    h_edges = np.linspace(0, 360, 37)   # 36 bins, 10 degrees each
    s_edges = np.linspace(0, 1, 21)
    v_edges = np.linspace(0, 1, 21)
    c_edges = np.linspace(0, 1, 21)

    h_histograms = []
    s_histograms = []
    v_histograms = []
    c_histograms = []

    for path in paths:
        try:
            with Image.open(path) as image:
                image = image.convert('RGB')
                h, s, v, c = rgb_to_hsvcr(image)

        except Exception as e:
            print(f'Could not process {path}: {e}')
            continue

        # Hue
        hue_mask = s > 0.05 # Arbitrary threshold - for very low saturation, hue becomes meaningless.

        if np.any(hue_mask):
            h_hist, _ = np.histogram(h[hue_mask], bins = h_edges)
            h_hist = h_hist.astype(np.float64)
            h_hist /= h_hist.sum()
            h_histograms.append(h_hist)

        # Saturation
        s_hist, _ = np.histogram(s, bins = s_edges)
        s_hist = s_hist.astype(np.float64)
        s_hist /= s_hist.sum()
        s_histograms.append(s_hist)

        # Value
        v_hist, _ = np.histogram(v, bins = v_edges)
        v_hist = v_hist.astype(np.float64)
        v_hist /= v_hist.sum()
        v_histograms.append(v_hist)

        # Chroma
        c_hist, _ = np.histogram(c, bins = c_edges)
        c_hist = c_hist.astype(np.float64)
        c_hist /= c_hist.sum()
        c_histograms.append(c_hist)

    stats = {'h': {'hist': np.mean(h_histograms, axis = 0) if h_histograms else None,
                   'edges': h_edges},
             's': {'hist': np.mean(s_histograms, axis = 0) if s_histograms else None,
                   'edges': s_edges},
             'v': {'hist': np.mean(v_histograms, axis = 0) if v_histograms else None,
                   'edges': v_edges},
             'c': {'hist': np.mean(c_histograms, axis = 0) if c_histograms else None,
                   'edges': c_edges}}

    return stats

def calculate_circular_mean(values: np.ndarray | list[float], weights: np.ndarray | list[float] | None = None) -> float:
    """
    Calculates the weighted circular mean for angular data (e.g., Hue).

    Converts angles to radians, computes the weighted average of their sine 
    and cosine components, and converts the resulting vector back to an angle.

    Args:
        values (np.ndarray | list[float]): Angular values in degrees [0, 360).
        weights (np.ndarray | list[float] | None, optional): Weights for each value. 
            Defaults to None (equal weighting).

    Returns:
        float: The circular mean in degrees [0, 360). Returns np.nan if the input 
            is empty, total weight is zero, or the resultant vector length is ~0.
    """

    if len(values) == 0:
        return np.nan

    values = np.asarray(values, dtype = np.float64)

    if weights is None:
        weights = np.ones_like(values)
    else:
        weights = np.asarray(weights, dtype = np.float64)

    total_weight = np.sum(weights)

    if total_weight == 0:
        return np.nan

    radians = np.deg2rad(values)

    sin_mean = np.sum(weights * np.sin(radians)) / total_weight
    cos_mean = np.sum(weights * np.cos(radians)) / total_weight

    r = np.hypot(sin_mean, cos_mean)

    if r < 1e-12:
        return np.nan

    angle = np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360

    if np.isclose(angle, 360.0, atol = 1e-12):
        angle = 0.0

    return angle

def calculate_circular_median(values: np.ndarray | list[float], weights: np.ndarray | list[float] | None = None) -> float:
    """
    Calculates the weighted circular median for angular data.

    Finds the angle in the given dataset that minimizes the sum of absolute 
    angular differences to all other points.

    Args:
        values (np.ndarray | list[float]): Angular values in degrees [0, 360).
        weights (np.ndarray | list[float] | None, optional): Weights for each value. 
            Defaults to None.

    Returns:
        float: The circular median in degrees. Returns np.nan if input is empty.
    """

    if len(values) == 0:
        return np.nan

    values = np.asarray(values, dtype = np.float64)

    if weights is None:
        weights = np.ones_like(values)
    else:
        weights = np.asarray(weights, dtype = np.float64)

    if np.sum(weights) == 0:
        return np.nan

    values = values % 360
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    candidate_angles = values
    distances = np.abs(candidate_angles[:, None] - values[None, :])
    distances = np.minimum(distances, 360 - distances)
    total_distances = np.sum(distances * weights[None, :], axis = 1)
    return float(candidate_angles[np.argmin(total_distances)])

def get_hist_mean(hist: np.ndarray | None, edges: np.ndarray) -> float:
    """
    Calculates the mean of a distribution represented by a histogram.

    Args:
        hist (np.ndarray | None): Histogram bin values/counts.
        edges (np.ndarray): The bin edges of the histogram.

    Returns:
        float: The weighted average of the bin centers. Returns np.nan if empty.
    """

    if hist is None or len(hist) == 0:
        return np.nan

    bin_centers = (edges[:-1] + edges[1:]) / 2
    return np.sum(hist * bin_centers) / np.sum(hist)

def get_hist_median(hist: np.ndarray | None, edges: np.ndarray) -> float:
    """
    Calculates the median of a distribution represented by a histogram.

    Uses the Cumulative Distribution Function (CDF) to interpolate the exact 
    median value within the target bin.

    Args:
        hist (np.ndarray | None): Histogram bin values/counts.
        edges (np.ndarray): The bin edges of the histogram.

    Returns:
        float: The interpolated median value. Returns np.nan if empty.
    """

    if hist is None or len(hist) == 0:
        return np.nan

    total = np.sum(hist)
    if total == 0:
        return np.nan

    cdf = np.cumsum(hist) / total
    median_bin = np.searchsorted(cdf, 0.5)
    median_bin = min(median_bin, len(hist) - 1)

    lower_edge = edges[median_bin]
    upper_edge = edges[median_bin + 1]

    previous_cdf = cdf[median_bin - 1] if median_bin > 0 else 0.0
    bin_probability = cdf[median_bin] - previous_cdf

    if bin_probability == 0:
        return (lower_edge + upper_edge) / 2

    fraction = (0.5 - previous_cdf) / bin_probability

    return float(lower_edge + fraction * (upper_edge - lower_edge))

def calculate_statistics(stats: dict[str, dict[str, Any]]) -> tuple[float, ...]:
    """
    Extracts and computes mean and median stats for all HSVCR channels.

    Args:
        stats (dict[str, dict[str, Any]]): Dictionary containing 'hist' and 
            'edges' for keys 'h', 's', 'v', 'c'.

    Returns:
        tuple[float, ...]: A tuple of 8 float values containing:
            (mean_h, median_h, mean_s, median_s, mean_v, median_v, mean_c, median_c).
    """

    # Hue requires circular statistics
    if stats['h']['hist'] is not None:
        h_bin_centers = (stats['h']['edges'][:-1] + stats['h']['edges'][1:]) / 2
        mean_h = calculate_circular_mean(h_bin_centers, stats['h']['hist'])
        median_h = calculate_circular_median(h_bin_centers, stats['h']['hist'])
    else:
        mean_h = np.nan
        median_h = np.nan

    # Saturation
    mean_s = get_hist_mean(stats['s']['hist'], stats['s']['edges'])
    median_s = get_hist_median(stats['s']['hist'], stats['s']['edges'])

    # Value
    mean_v = get_hist_mean(stats['v']['hist'], stats['v']['edges'])
    median_v = get_hist_median(stats['v']['hist'], stats['v']['edges'])

    # Chroma
    mean_c = get_hist_mean(stats['c']['hist'], stats['c']['edges'])
    median_c = get_hist_median(stats['c']['hist'], stats['c']['edges'])

    return mean_h, median_h, mean_s, median_s, mean_v, median_v, mean_c, median_c

def get_color_statistics(paths_dict: dict[str, list[Path]]) -> dict[str, dict[str, Any]]:
    """
    Calculates and prints color distribution statistics for multiple classes.

    Iterates through the provided classes, builds color histograms using their
    image paths, calculates the mean and median for each channel, and prints 
    the results to the console.

    Args:
        paths_dict (dict[str, list[Path]]): Dictionary mapping class names to 
            their respective lists of image paths.

    Returns:
        dict[str, dict[str, Any]]: The compiled dictionary mapping class names 
            to their 'h', 's', 'v', 'c' histogram data.
    """

    stats = {}

    for class_name, paths in paths_dict.items():
        print(f'Processing class: {class_name} ({len(paths)} images)')
        stats[class_name] = get_class_color_histograms(paths)
        add_stats = calculate_statistics(stats[class_name])
        print(f'--- {class_name} ---\n'
              f'Circular Mean Hue: {add_stats[0]:.2f}, Median Hue: {add_stats[1]:.2f}\n'
              f'Mean Saturation: {add_stats[2]:.2f}, Median Saturation: {add_stats[3]:.2f}\n'
              f'Mean Value: {add_stats[4]:.2f}, Median Value: {add_stats[5]:.2f}\n'
              f'Mean Chroma: {add_stats[6]:.2f}, Median Chroma: {add_stats[7]:.2f}\n')

    return stats

def classification_eda(path: str) -> None:
    """
    Performs Exploratory Data Analysis (EDA) for image classification.

    Extracts class paths, computes color statistics (Hue, Saturation, 
    Value, and Chroma), and generates heatmaps to visualize their distributions
    across different classes.

    Args:
        path (str): Path to the root directory containing subfolders for each class.
    """

    paths_dict = get_classes_paths(path)
    stats = get_color_statistics(paths_dict) 

    plot_configs = {'h': {'title': 'Hue Distribution by Class', 'xlabel': 'Hue (degrees)'},
                    's': {'title': 'Saturation Distribution by Class', 'xlabel': 'Saturation'},
                    'v': {'title': 'Value Distribution by Class', 'xlabel': 'Value'},
                    'c': {'title': 'Chroma Distribution by Class', 'xlabel': 'Chroma'}}

    for channel, config in plot_configs.items():
        plot_heatmap(stats, channel, config)