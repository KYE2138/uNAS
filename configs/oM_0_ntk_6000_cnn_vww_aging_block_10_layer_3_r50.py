import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from config import TrainingConfig, AgingEvoConfig, BoundConfig, PruningConfig, ThresholdConfig
from dataset import VisualWakeWords_50
from cnn import CnnSearchSpace
from search_algorithms import AgingEvoSearch

search_algorithm = AgingEvoSearch

training_config = TrainingConfig(
    dataset=VisualWakeWords_50("/docker/file/dataset/visualwakewords"),
    optimizer=lambda: tfa.optimizers.SGDW(learning_rate=0.001, weight_decay=5e-5),
    batch_size=16,
    epochs=30,
    callbacks=lambda: [EarlyStopping(patience=15, verbose=1)]
)

search_config = AgingEvoConfig(
    search_space=CnnSearchSpace(dropout=0.15),
    rounds=2000,
    checkpoint_dir="artifacts/cnn_vww",
    max_parallel_evaluations = 2
)

bound_config = BoundConfig(
    error_bound=0.20,
    peak_mem_bound=250000,
    model_size_bound=700000,
    mac_bound=80000000,
    ntk=6000,
    #rn=1500,
    #ntk_PMU = 200000000,
    #ntk_MS = 200000000,
    #ntk_MACs = 240000000000,
    #rn_PMU = 75000000,
    #rn_MS = 75000000,
    #rn_MACs = 90000000000
    #ntk_rn_PMU = 300000000000,
    #ntk_rn_MS = 300000000000,
    #ntk_rn_MACs = 360000000000000,
    #ntk_rn = 6000000,
    #PMU_d_orn = 30,
    #MS_d_orn = 60,
    #MACs_d_orn = 8000
)

threshold_config = ThresholdConfig(
    # bound_config.ntk * 3
    ntk = 18000,
    # 4000 - bound_config.rn
    #rn = 2500
)
