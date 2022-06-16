import tensorflow_addons as tfa

from cnn import CnnSearchSpace
from config import AgingEvoConfig, TrainingConfig, BoundConfig
from dataset import MNIST
from search_algorithms import AgingEvoSearch

search_algorithm = AgingEvoSearch

training_config = TrainingConfig(
    dataset=MNIST(),
    epochs=30,
    batch_size=64,
    optimizer=lambda: tfa.optimizers.SGDW(learning_rate=0.005, momentum=0.9, weight_decay=4e-5),
    callbacks=lambda: [],
)

search_config = AgingEvoConfig(
    search_space=CnnSearchSpace(),
    checkpoint_dir="artifacts/cnn_mnist"
)

bound_config = BoundConfig(
    #error_bound=0.035,
    #peak_mem_bound=2500,
    #model_size_bound=4500,
    #mac_bound=30000000,
    ntk=1000,
    rn=1500,
    #ntk_PMU = 200000000,
    #ntk_MS = 200000000,
    #ntk_MACs = 240000000000,
    #rn_PMU = 75000000,
    #rn_MS = 75000000,
    #rn_MACs = 90000000000
    ntk_rn_PMU = 3750000000,
    ntk_rn_MS = 6750000000,
    ntk_rn_MACs = 45000000000000,
    ntk_rn = 1500000
)


from config import PruningConfig

training_config.pruning = PruningConfig(
    structured=True,
    start_pruning_at_epoch=3,
    finish_pruning_by_epoch=18,
    min_sparsity=0.05,
    max_sparsity=0.8
)
