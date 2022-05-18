import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

from config import TrainingConfig, AgingEvoConfig, BoundConfig
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
    batch_size=128,
    #epochs=130,
    epochs=1,
    callbacks=lambda: [LearningRateScheduler(lr_schedule)],
)


search_config = AgingEvoConfig(
    search_space=CnnSearchSpace(dropout=0.15),
    rounds=6000,
    checkpoint_dir="artifacts/cnn_cifar10",
    max_parallel_evaluations = 1,
    population_size = 10,
    sample_size = 5
)

bound_config = BoundConfig(
    error_bound=0.18,
    peak_mem_bound=75000,
    model_size_bound=75000,
    mac_bound=30000000,
    ntk=4000
)
