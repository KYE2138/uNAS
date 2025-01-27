import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from config import TrainingConfig, AgingEvoConfig, BoundConfig, PruningConfig
from dataset import VisualWakeWords
from cnn import CnnSearchSpace
from search_algorithms import AgingEvoSearch

search_algorithm = AgingEvoSearch

training_config = TrainingConfig(
    dataset=VisualWakeWords("/docker/file/dataset/visualwakewords"),
    optimizer=lambda: tfa.optimizers.SGDW(learning_rate=0.001, weight_decay=5e-5),
    batch_size=32,
    epochs=35,
    callbacks=lambda: []
    #callbacks=lambda: [EarlyStopping(patience=15, verbose=1)]
)

training_config.pruning = PruningConfig(
    structured=True,
    start_pruning_at_epoch=12,
    finish_pruning_by_epoch=28,
    min_sparsity=0.1,
    max_sparsity=0.90
)

search_config = AgingEvoConfig(
    search_space=CnnSearchSpace(),
    rounds=6000,
    checkpoint_dir="artifacts/cnn_vww",
    max_parallel_evaluations = 1,
    population_size = 100,
    sample_size = 25
)

bound_config = BoundConfig(
    error_bound=0.10,
    peak_mem_bound=250000,
    model_size_bound=325000,
    mac_bound=60000000,
    ntk=5000
)
