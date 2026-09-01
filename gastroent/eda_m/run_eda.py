from eda_m.classification import classification_eda
from eda_m.segmentation import segmentation_eda

def run_classification_eda() -> None:
    """
    Runs EDA on all classification datasets.
    """

    classification_eda('data/raw/Kvasir')
    classification_eda('data/raw/HyperKvasir')

def run_segmentation_eda() -> None:
    """
    Runs EDA on all segmentation datasets. Includes polyp ratio and HSV statistics.
    """
    
    segmentation_eda('data/raw/CVC-ClinicDB/masks')
    segmentation_eda('data/raw/Kvasir-SEG/kvasir-seg/masks')
    segmentation_eda('data/raw/Kvasir-SEG/kvasir-sessile/masks')
    classification_eda('data/raw/CVC-ClinicDB')
    classification_eda('data/raw/Kvasir-SEG')