import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from config import TrainingConfig, AgingEvoConfig, BoundConfig, PruningConfig
from dataset import VisualWakeWords
from cnn import CnnSearchSpace
from search_algorithms import AgingEvoSearch

search_algorithm = AgingEvoSearch

training_config = TrainingConfig(
    dataset=VisualWakeWords("/storage/dataset/visualwakewords"),
    optimizer=lambda: tfa.optimizers.SGDW(learning_rate=0.001, weight_decay=5e-5),
    batch_size=32,
    epochs=30,
    #callbacks=lambda: [EarlyStopping(patience=15, verbose=1)]
)

search_config = AgingEvoConfig(
    search_space=CnnSearchSpace(),
    rounds=2000,
    checkpoint_dir="artifacts/cnn_vww",
    max_parallel_evaluations = 1
)

bound_config = BoundConfig(
    error_bound=0.10,
    peak_mem_bound=250000,
    model_size_bound=250000,
    mac_bound=30000000,
    ntk=5000
)
