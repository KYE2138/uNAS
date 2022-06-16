import os
import argparse
import logging
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

from pathlib import Path

from search_algorithms import BayesOpt


def main():
    # 設定log
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log = logging.getLogger("Driver")
    # 取得參數
    parser = argparse.ArgumentParser("uNAS Search")
    parser.add_argument("config_file", type=str, help="A config file describing the search parameters")
    parser.add_argument("--name", type=str, help="Experiment name (for disambiguation during state saving)")
    parser.add_argument("--load-from", type=str, default=None, help="A search state file to resume from")
    parser.add_argument("--save-every", type=int, default=5, help="After how many search steps to save the state")
    parser.add_argument("--seed", type=int, default=0, help="A seed for the global NumPy and TensorFlow random state")
    args = parser.parse_args()
    
    # 設定隨機變數seed
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # gpu相關
    '''
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    '''
    '''
    # limit gpu mem to load keras model and transfer
    gpus = tf.config.list_physical_devices('GPU')
    print (gpus)
    if gpus:
        # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    '''

    #del metrics
    save_path = './tmp/metrics/ntk_rn'
    train_loader_save_path = f'{save_path}/train_loader.pickle'
    val_loader_save_path = f'{save_path}/val_loader.pickle'
    if os.path.isfile(train_loader_save_path) and os.path.isfile(val_loader_save_path):
        print (f"train_loader_save_path is already exist:{train_loader_save_path}")
        print (f"val_loader_save_path is already exist:{val_loader_save_path}")
        os.remove(train_loader_save_path)
        os.remove(val_loader_save_path)


    # 檢查參數
    if args.save_every <= 0:
        raise argparse.ArgumentTypeError("Value for '--save-every' must be a positive integer.")
    
    # 執行config_file(.py)內之code,configs 則是全域變數(以字典型態儲存)
    configs = {}
    exec(Path(args.config_file).read_text(), configs)
    
    # 執行完config_file後, algo會等於uNAS/search_algorithms下之.py中的class, 如algo = AgingEvoSearch
    # 若未設定search_algorithm參數, 則將algo設定為BayesOpt
    if "search_algorithm" not in configs:
        algo = BayesOpt
    else:
        algo = configs["search_algorithm"]

    
    # 獲取config_file內之參數值
    search_space = configs["search_config"].search_space
    dataset = configs["training_config"].dataset
    search_space.input_shape = dataset.input_shape
    search_space.num_classes = dataset.num_classes
    
    # 設定搜尋演算法, algo(class)為uNAS/search_algorithms下之.py中的class, 如AgingEvoSearch
    search = algo(experiment_name=args.name or "search",
                  search_config=configs["search_config"],
                  training_config=configs["training_config"],
                  bound_config=configs["bound_config"],
                  threshold_config=configs["threshold_config"])
    # 開始搜尋, search為uNAS/search_algorithms下之.py中的class的method
    if args.load_from and not os.path.exists(args.load_from):
        log.warning("Search state file to load from is not found, the search will start from scratch.")
        args.load_from = None
    search.search(load_from=args.load_from, save_every=args.save_every)


if __name__ == "__main__":
    main()
