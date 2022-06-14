from cgi import parse_multipart
import ray
import logging
import pickle
import numpy as np
import tensorflow as tf
from pathlib import Path
from dataclasses import dataclass
from typing import List, Union, Optional

from architecture import Architecture
from config import AgingEvoConfig, TrainingConfig, BoundConfig
from model_trainer import ModelTrainer
from resource_models.models import peak_memory_usage, model_size, inference_latency
from utils import Scheduler, debug_mode

#ntk
#from ntk import ModelNTK
#from metrics_file import ModelMetricsFile
from metrics_file_ntk_rn import ModelMetricsFile

import pdb
import gc

@dataclass
class ArchitecturePoint:
    arch: Architecture
    sparsity: Optional[float] = None


@dataclass
class EvaluatedPoint:
    point: ArchitecturePoint
    val_error: float
    test_error: float
    resource_features: List[Union[int, float]]


@ray.remote(num_gpus=0 if debug_mode() else 1, num_cpus=1 if debug_mode() else 6)
class GPUTrainer:
    def __init__(self, search_space, trainer, bound_config):
        self.trainer = trainer
        self.ss = search_space
        self.bound_config = bound_config
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")



    #在utils.py內有worker.evaluate.remote，在ray下執行
    def evaluate(self, point):
        log = logging.getLogger("Worker")

        data = self.trainer.dataset
        arch = point.arch
        print (f'model input_shape:{data.input_shape}')
        print (f'model_rn input_shape:(2, 2, 1)')
        model = self.ss.to_keras_model(arch, data.input_shape, data.num_classes)
        model_rn = self.ss.to_keras_model(arch, (2, 2, 1), data.num_classes)
        
        # pre ntk
        #metrics_file.py
        #ntks = ModelMetricsFile(self.trainer).get_metrics(model, num_batch=1)
        #metrics_file_ntk_rn.py
        ntks, rns= ModelMetricsFile(self.trainer).get_metrics(model=model, model_rn=model_rn, num_batch=1, num_networks=3)
        #pdb.set_trace()

        ntk = np.mean(ntks).astype('int64')
        # ntk_threshold是bound三倍
        ntk_threshold = int(self.bound_config.ntk)*3
        # rns
        rn = np.mean(rns).astype('int64')
        #max rn ~ 3000
        # 限制rn在1500以上
        rn = 4000-rn
        # rn要在2500以下
        #pdb.set_trace()
        if ntk<0 or ntk>ntk_threshold or rn>=2500:
            print(f'ntks = {ntks}, ntk = {ntk}')
            print(f'rns = {rns}, rn = {rn}')
            print(f'epochs = 1')
            results = self.trainer.train_and_eval(model, sparsity=point.sparsity, epochs=1)
        else:
            print(f'ntks = {ntks}, ntk = {ntk}')
            print(f'rns = {rns}, rn = {rn}')
            print(f'epochs = {self.trainer.config.epochs}')
            results = self.trainer.train_and_eval(model, sparsity=point.sparsity)
        
        
        #使用model_trainer.py內的ModelTrainer類別中的train_and_eval函數
        #results = self.trainer.train_and_eval(model, sparsity=point.sparsity)
        val_error, test_error = results["val_error"], results["test_error"]
        rg = self.ss.to_resource_graph(arch, data.input_shape, data.num_classes,
                                       pruned_weights=results["pruned_weights"])
        unstructured_sparsity = self.trainer.config.pruning and \
                                not self.trainer.config.pruning.structured
        resource_features = [peak_memory_usage(rg), model_size(rg, sparse=unstructured_sparsity),
                             inference_latency(rg, compute_weight=1, mem_access_weight=0)]
        # resource_features = [175104, 164176, 61449631]
        '''
        #ntk
        # data save as numpy
        # modle save as keras model
        ntks = ModelMetricsFile(self.trainer).get_metrics(model, num_batch=1)
        ntk = np.mean(ntks).astype('int64')
        '''
        resource_features.append(ntk)
        resource_features.append(rn)
        # get lagecy metrics
        PMU=resource_features[0]
        MS=resource_features[1]
        MACs=resource_features[2]
        # new metrics
        positive_ntk = ntk
        if positive_ntk<0:
            positive_ntk=ntk_threshold+1
        ntk_PMU=positive_ntk*PMU
        ntk_MS=positive_ntk*MS
        ntk_MACs=positive_ntk*MACs
        resource_features.append(ntk_PMU)
        resource_features.append(ntk_MS)
        resource_features.append(ntk_MACs)
        
        rn_PMU=rn*PMU
        rn_MS=rn*MS
        rn_MACs=rn*MACs
        resource_features.append(rn_PMU)
        resource_features.append(rn_MS)
        resource_features.append(rn_MACs)
        
        print(f'ntk_PMU = {ntk_PMU}, ntk_MS = {ntk_MS}, ntk_MACs = {ntk_MACs}')
        print(f'rn_PMU = {rn_PMU}, rn_MS = {rn_MS}, rn_MACs = {rn_MACs}')



        #pdb.set_trace()
        
        log.info(f"Training complete: val_error={val_error:.4f}, test_error={test_error:.4f}, "
                 f"resource_features={resource_features}.")
        return EvaluatedPoint(point=point,
                              val_error=val_error, test_error=test_error,
                              resource_features=resource_features)


class AgingEvoSearch:
    def __init__(self,
                 experiment_name: str,
                 search_config: AgingEvoConfig,
                 training_config: TrainingConfig,
                 bound_config: BoundConfig):
        self.log = logging.getLogger(name=f"AgingEvoSearch [{experiment_name}]")
        self.config = search_config
        #引用model_trainer.py內的ModelTrainer Class
        self.trainer = ModelTrainer(training_config)
        self.root_dir = Path(search_config.checkpoint_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.bound_config = bound_config
        if training_config.pruning and not training_config.pruning.structured:
            self.log.warning("For unstructured pruning, we can only meaningfully use the model "
                             "size resource metric.")
            bound_config.peak_mem_bound = None
            bound_config.mac_bound = None
        self.pruning = training_config.pruning

        # We establish an order of objective in the feature vector, all functions must ensure the order is the same
        self.constraint_bounds = [bound_config.error_bound,
                                  bound_config.peak_mem_bound,
                                  bound_config.model_size_bound,
                                  bound_config.mac_bound,
                                  bound_config.ntk]

        self.history: List[EvaluatedPoint] = []
        self.population: List[EvaluatedPoint] = []

        self.population_size = search_config.population_size
        self.initial_population_size = search_config.initial_population_size or self.population_size
        self.rounds = search_config.rounds
        self.sample_size = search_config.sample_size
        num_gpus = len(tf.config.experimental.list_physical_devices("GPU"))
        self.max_parallel_evaluations = search_config.max_parallel_evaluations or num_gpus

    def save_state(self, file):
        with open(file, "wb") as f:
            pickle.dump(self.history, f)
        self.log.info(f"Saved {len(self.history)} architectures to {file}.")

    def load_state(self, file):
        with open(file, "rb") as f:
            self.history = pickle.load(f)
        self.population = self.history[-self.population_size:]
        self.log.info(f"Loaded {len(self.history)} architectures from {file}")

    def maybe_save_state(self, save_every):
        if len(self.history) % save_every == 0:
            file = self.root_dir / f"{self.experiment_name}_agingevosearch_state.pickle"
            self.save_state(file.as_posix())

    def get_mo_fitness_fn(self):
        lambdas = np.random.uniform(low=0.0, high=1.0, size=6)

        def normalise(x, l=0, u=1, cap=10.0):
            return min((x - l) / (u - l), cap)

        def fitness(i: EvaluatedPoint):
            #resource_features = [peak_memory_usage(rg), model_size(rg, sparse=unstructured_sparsity), inference_latency(rg, compute_weight=1, mem_access_weight=0)]
            #fetures = [0.6031999886035919, 40404, 6973, 651048]
            features = [i.val_error] + i.resource_features

            # All objectives must be non-negative and scaled to the same magnitude of
            # between 0 and 1. Values that exceed required bounds will therefore be mapped
            # to a factor > 1, and be hit by the optimiser first.
            # 所有目標都必須是非負的，並且縮放到 0 到 1 之間的相同大小。因此，超出所需範圍的值將被映射到大於 1 的因子，並首先被優化器擊中。
            # fetures = [0.6031999886035919, 40404, 6973, 651048]
            # self.constraint_bounds = [0.1, 50000, 50000, 60000000]
            # lambdas = array([0.69199057, 0.25018253, 0.38504929, 0.56419586])
            # normalised_features = [8.716881661420613, 3.2299617023466287, 0.36218739220099627, 0.019232328286649242]
            normalised_features = [normalise(f, u=c) / l
                                   for f, c, l in zip(features, self.constraint_bounds, lambdas)
                                   if c is not None]  # bound = None means ignored objective
            
            # -max(normalised_features) = -8.716881661420613
            return -max(normalised_features)  # Negated, due to function being maximised
        return fitness

    def bounds_log(self, history_size=25):
        def to_feature_vector(i):
            return [i.val_error] + i.resource_features
        within_bounds = \
            [all(o <= b
                 for o, b in zip(to_feature_vector(i), self.constraint_bounds)
                 if b is not None)
             for i in self.history[-history_size:]]
        self.log.info(f"In bounds: {sum(within_bounds)} within "
                      f"the last {len(within_bounds)} architectures.")

    def evolve(self, point: ArchitecturePoint):
        arch = np.random.choice(self.config.search_space.produce_morphs(point.arch))
        sparsity = None
        if self.pruning:
            incr = np.random.normal(loc=0.0, scale=0.05)
            sparsity = np.clip(point.sparsity + incr,
                               self.pruning.min_sparsity, self.pruning.max_sparsity)
        return ArchitecturePoint(arch=arch, sparsity=sparsity)

    def random_sample(self):
        arch = self.config.search_space.random_architecture()
        sparsity = None
        if self.pruning:
            sparsity = np.random.uniform(self.pruning.min_sparsity, self.pruning.max_sparsity)
        return ArchitecturePoint(arch=arch, sparsity=sparsity)

    def search(self, load_from: str = None, save_every: int = None):
        if load_from:
            self.load_state(load_from)

        ray.init(local_mode=debug_mode())

        trainer = ray.put(self.trainer)
        ss = ray.put(self.config.search_space)
        bound = ray.put(self.bound_config)
        #utils.py內的Scheduler class
        #Create some GpuTrainer actors
        #建立一個list，每個元素內，建立一個GpuTrainer actors，再傳入Scheduler Class做初始化
        scheduler = Scheduler([GPUTrainer.remote(ss, trainer, bound)
                               for _ in range(self.max_parallel_evaluations)])
        self.log.info(f"Searching with {self.max_parallel_evaluations} workers.")

        def should_submit_more(cap):
            return (len(self.history) + scheduler.pending_tasks() < cap) \
               and scheduler.has_a_free_worker()

        def point_number():
            return len(self.history) + scheduler.pending_tasks() + 1

        while len(self.history) < self.initial_population_size:
            if should_submit_more(cap=self.initial_population_size):
                self.log.info(f"Populating #{point_number()}...")
                scheduler.submit(self.random_sample())
            else:
                info = scheduler.await_any()
                self.population.append(info)
                self.history.append(info)
                self.maybe_save_state(save_every)

        while len(self.history) < self.rounds:
            if should_submit_more(cap=self.rounds):
                self.log.info(f"Searching #{point_number()}...")
                # 從population中隨機選擇幾個(sample_size)架構
                sample = np.random.choice(self.population, size=self.sample_size)
                # 對幾個隨機選出的架構，目標函數輸出值最大的做為parent
                parent = max(sample, key=self.get_mo_fitness_fn())

                scheduler.submit(self.evolve(parent.point))
            else:
                info = scheduler.await_any()
                self.population.append(info)
                while len(self.population) > self.population_size:
                    self.population.pop(0)
                self.history.append(info)
                self.maybe_save_state(save_every)
                self.bounds_log()
