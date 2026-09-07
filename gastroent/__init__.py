import torch

from gastroent.main import main
from gastroent.utils_m.utils import set_seeds

SEED = 42

if __name__ == '__main__':
    print('CUDA Available:', torch.cuda.is_available())
    print('GPU Count:', torch.cuda.device_count())
    set_seeds(SEED)
    main(SEED)