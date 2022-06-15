
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

from config import TrainingConfig, AgingEvoConfig, BoundConfig, PruningConfig
from dataset import CIFAR10
from cnn import CnnSearchSpace
from search_algorithms import AgingEvoSearch


def lr_schedule(epoch):
    if 0 <= epoch < 90:
        return 0.01
    if 90 <= epoch < 105:
        return 0.005
    return 0.001


search_algorithm = AgingEvoSearch

training_config = TrainingConfig(
    dataset=CIFAR10(),
    optimizer=lambda: tfa.optimizers.SGDW(learning_rate=0.01, momentum=0.9, weight_decay=1e-5),
    batch_size=64,
    epochs=130,
    #epochs=1,
    callbacks=lambda: [LearningRateScheduler(lr_schedule)],
)


search_config = AgingEvoConfig(
    search_space=CnnSearchSpace(dropout=0.15),
    rounds=6000,
    checkpoint_dir="artifacts/cnn_cifar10",
    max_parallel_evaluations = 1,
    #population_size = 10,
    #sample_size = 5
)

training_config.pruning = PruningConfig(
    structured=True,
    start_pruning_at_epoch=90,
    finish_pruning_by_epoch=120,
    min_sparsity=0.1,
    max_sparsity=0.90
)

bound_config = BoundConfig(
    #error_bound=0.15,
    #peak_mem_bound=50000,
    #model_size_bound=50000,
    #mac_bound=60000000,
    ntk=4000,
    rn=1500,
    #ntk_PMU = 200000000,
    #ntk_MS = 200000000,
    #ntk_MACs = 240000000000,
    #rn_PMU = 75000000,
    #rn_MS = 75000000,
    #rn_MACs = 90000000000
    ntk_rn_PMU = 300000000000,
    ntk_rn_MS = 300000000000,
    ntk_rn_MACs = 360000000000000,
)
