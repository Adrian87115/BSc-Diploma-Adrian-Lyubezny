from gastroent.main import main
from gastroent.utils import set_seeds

SEED = 42

if __name__ == "__main__":
    set_seeds(SEED)
    main(SEED)