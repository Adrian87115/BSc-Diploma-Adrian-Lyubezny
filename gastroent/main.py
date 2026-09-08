from models_m.classification_models import ResNetModel, ConvNeXtModel, SwinTransformerModel
from models_m.segmentation_models import UNetPlusPlusModel, DeepLabV3PlusModel, SegFormerModel
from utils_m.experiments import classification_experiment, segmentation_experiment
from eda_m.run_eda import run_classification_eda, run_segmentation_eda
from utils_m.tests import test_classification, test_segmentation

# GENERIC CLASSES FROM KVASIR MAY CONFUSE THE SPECIFIC CLASSES IN HYPERKVASIR. 
# THIS IS NOT AN ISSUE IF DATASETS ARE NOT MERGED OR CROSS-TESTED.

# TO DO:
# augmentations
# DATALOADER IS REPSONSOBLE FOR THE OVERSAMPLING/UNDERSAMPLING  WeightedRandomSamplerXXXDistributedSampler

def main(seed):
    # run_classification_eda()
    # run_segmentation_eda()

    classification_experiment(seed, 'Kvasir1')
    # segmentation_experiment(seed, 'KvasirSeg1')

    # test_classification(ResNetModel(3, 24, '18'), 'Kvasir1', 2, 20, 'HyperKvasir/lower-gi-tract/pathological-findings/polyps/2dadc75e-8fca-4411-88a0-65a3f1cc92be.jpg', labels = 'polyps')
    # test_segmentation(DeepLabV3PlusModel(3, 'resnet18'), 'KvasirSeg1', 3, 20, 'CVC-ClinicDB', ['3.png', '26.png'])