# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Standard library imports
import logging
import os
import tempfile
import time
import uuid
from datetime import datetime
from typing import Any, List, NamedTuple, Optional, Union

# Third-party imports
import mxnet as mx
import mxnet.autograd as autograd
import mxnet.gluon.nn as nn
import numpy as np
from mxboard import *

# First-party imports
from gluonts.core.component import get_mxnet_context, validated
from gluonts.core.exception import GluonTSDataError, GluonTSUserError
from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader
from gluonts.support.util import HybridContext
from gluonts.gluonts_tqdm import tqdm

from gluonts.trainer import learning_rate_scheduler as lrs, Trainer


logger = logging.getLogger("gluonts").getChild("trainer")


MODEL_ARTIFACT_FILE_NAME = "model"
STATE_ARTIFACT_FILE_NAME = "state"

# make the IDE happy: mx.py does not explicitly import autograd
mx.autograd = autograd


def check_loss_finite(val: float) -> None:
    if not np.isfinite(val):
        raise GluonTSDataError(
            "Encountered invalid loss value! Try reducing the learning rate "
            "or try a different likelihood."
        )


def loss_value(loss: mx.metric.Loss) -> float:
    return loss.get_name_value()[0][1]


class BestEpochInfo(NamedTuple):
    params_path: str
    epoch_no: int
    metric_value: float


class TrackingTrainer(Trainer):
    """Extends gluonts.trainer.Trainer by adding the ability to track gradients, weights
    loss, etc., during training with MXBoard."""

    def __init__(self,
                 model_name: str,
                 dataset_name: str,
                 weight_seed: int = None,
                 batch_seed: int = None,
                 ctx: Optional[mx.Context] = None,
                 epochs: int = 100,
                 batch_size: int = 32,
                 num_batches_per_epoch: int = 50,
                 learning_rate: float = 1e-3,
                 learning_rate_decay_factor: float = 0.5,
                 patience: int = 10,
                 minimum_learning_rate: float = 5e-5,
                 clip_gradient: float = 10.0,
                 weight_decay: float = 1e-8,
                 init: Union[str, mx.initializer.Initializer] = "xavier",
                 hybridize: bool = True,
                ) -> None:
        super().__init__(
            ctx,
            epochs,
            batch_size,
            num_batches_per_epoch,
            learning_rate,
            learning_rate_decay_factor,
            patience,
            minimum_learning_rate,
            clip_gradient,
            weight_decay,
            init,
            hybridize
        )
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.weight_seed = weight_seed
        self.batch_seed = batch_seed

    def __call__(self,
                 net: nn.HybridBlock,
                 input_names: List[str],
                 train_iter: TrainDataLoader,
                 initialize: bool = False,
                 validation_iter: Optional[ValidationDataLoader] = None,
                 ) -> None:
        is_validation_available = validation_iter is not None
        self.halt = False

        current_time = datetime.now().strftime("%Y-%m-%d")
        sw = SummaryWriter(
            logdir=f"./logs/{self.model_name}/{self.dataset_name}/weightseed{self.weight_seed}batchseed{self.batch_seed}_{current_time}",
            flush_secs=10,
            verbose=False
        )

        with tempfile.TemporaryDirectory(
            prefix="gluonts-trainer-temp-"
        ) as gluonts_temp:

            def base_path() -> str:
                return os.path.join(
                    gluonts_temp,
                    "{}_{}".format(STATE_ARTIFACT_FILE_NAME, uuid.uuid4()),
                )

            logger.info("Start model training")

            net.initialize(ctx=self.ctx, init=self.init)

            with HybridContext(
                net=net,
                hybridize=self.hybridize,
                static_alloc=True,
                static_shape=True,
            ):
                batch_size = train_iter.batch_size

                best_epoch_info = BestEpochInfo(
                    params_path="%s-%s.params" % (base_path(), "init"),
                    epoch_no=-1,
                    metric_value=np.Inf,
                )

                lr_scheduler = lrs.MetricAttentiveScheduler(
                    objective="min",
                    patience=self.patience,
                    decay_factor=self.learning_rate_decay_factor,
                    min_lr=self.minimum_learning_rate,
                )

                optimizer = mx.optimizer.Adam(
                    learning_rate=self.learning_rate,
                    lr_scheduler=lr_scheduler,
                    wd=self.weight_decay,
                    clip_gradient=self.clip_gradient,
                )

                trainer = mx.gluon.Trainer(
                    net.collect_params(),
                    optimizer=optimizer,
                    kvstore="device",  # FIXME: initialize properly
                )

                def loop(
                    epoch_no, batch_iter, is_training: bool = True
                ) -> mx.metric.Loss:
                    tic = time.time()

                    epoch_loss = mx.metric.Loss()

                    with tqdm(batch_iter) as it:
                        for batch_no, data_entry in enumerate(it, start=1):
                            if self.halt:
                                break

                            inputs = [data_entry[k] for k in input_names]

                            with mx.autograd.record():
                                output = net(*inputs)

                                # network can returns several outputs, the first being always the loss
                                # when having multiple outputs, the forward returns a list in the case of hybrid and a
                                # tuple otherwise
                                # we may wrap network outputs in the future to avoid this type check
                                if isinstance(output, (list, tuple)):
                                    loss = output[0]
                                else:
                                    loss = output

                            if is_training:
                                loss.backward()
                                trainer.step(batch_size)

                            epoch_loss.update(None, preds=loss)
                            lv = loss_value(epoch_loss)

                            if not np.isfinite(lv):
                                logger.warning(
                                    "Epoch[%d] gave nan loss", epoch_no
                                )
                                return epoch_loss

                            it.set_postfix(
                                ordered_dict={
                                    "epoch": f"{epoch_no + 1}/{self.epochs}",
                                    ("" if is_training else "validation_")
                                    + "avg_epoch_loss": lv,
                                },
                                refresh=False,
                            )
                            # print out parameters of the network at the first pass
                            if batch_no == 1 and epoch_no == 0:
                                net_name = type(net).__name__
                                num_model_param = self.count_model_params(net)
                                logger.info(
                                    f"Number of parameters in {net_name}: {num_model_param}"
                                )
                    # mark epoch end time and log time cost of current epoch
                    toc = time.time()
                    logger.info(
                        "Epoch[%d] Elapsed time %.3f seconds",
                        epoch_no,
                        (toc - tic),
                    )

                    logger.info(
                        "Epoch[%d] Evaluation metric '%s'=%f",
                        epoch_no,
                        ("" if is_training else "validation_") + "epoch_loss",
                        lv,
                    )
                    return epoch_loss

                # Set the seed for the intial batch sampling
                if self.batch_seed is not None:
                    mx.random.seed(self.batch_seed)
                    np.random.seed(self.batch_seed)

                # Initialize the weights
                if self.weight_seed is not None:
                    data_entry = next(iter(train_iter))
                    inputs = [data_entry[k] for k in input_names]
                    _ = net(*inputs)

                    mx.random.seed(self.weight_seed)
                    np.random.seed(self.weight_seed)
                    net.initialize(ctx=self.ctx, init=self.init, force_reinit=True)
                else:
                    mx.random.seed(int(time.time()))
                    np.random.seed(int(time.time()))

                # Set the seed for batch sampling
                if self.batch_seed is not None:
                    mx.random.seed(self.batch_seed)
                    np.random.seed(self.batch_seed)
                else:
                    mx.random.seed(int(time.time()))
                    np.random.seed(int(time.time()))

                for epoch_no in range(self.epochs):
                    if self.halt:
                        logger.info(f"Epoch[{epoch_no}] Interrupting training")
                        break

                    curr_lr = trainer.learning_rate
                    logger.info(
                        f"Epoch[{epoch_no}] Learning rate is {curr_lr}"
                    )

                    epoch_loss = loop(epoch_no, train_iter)
                    train_loss = loss_value(epoch_loss)
                    if is_validation_available:
                        epoch_loss = loop(
                            epoch_no, validation_iter, is_training=False
                        )
                        val_loss = loss_value(epoch_loss)


                    should_continue = lr_scheduler.step(loss_value(epoch_loss))
                    if not should_continue:
                        logger.info("Stopping training")
                        break

                    if loss_value(epoch_loss) < best_epoch_info.metric_value:
                        best_epoch_info = BestEpochInfo(
                            params_path="%s-%04d.params"
                            % (base_path(), epoch_no),
                            epoch_no=epoch_no,
                            metric_value=loss_value(epoch_loss),
                        )
                        net.save_parameters(
                            best_epoch_info.params_path
                        )  # TODO: handle possible exception

                    if not trainer.learning_rate == curr_lr:
                        if best_epoch_info.epoch_no == -1:
                            raise GluonTSUserError(
                                "Got NaN in first epoch. Try reducing initial learning rate."
                            )

                        logger.info(
                            f"Loading parameters from best epoch "
                            f"({best_epoch_info.epoch_no})"
                        )
                        net.load_parameters(
                            best_epoch_info.params_path, self.ctx
                        )
                    
                    # add loss to tensorboard
                    sw.add_scalar(
                        tag="training_loss",
                        value=train_loss,
                        global_step=epoch_no
                    )
                    
                    if is_validation_available:
                        sw.add_scalar(
                            tag="validation_loss",
                            value=val_loss,
                            global_step=epoch_no
                        )

                    # track the learning rate
                    sw.add_scalar(
                        tag="learning_rate",
                        value=trainer.optimizer.learning_rate,
                        global_step=epoch_no
                    )

                logger.info(
                    f"Loading parameters from best epoch "
                    f"({best_epoch_info.epoch_no})"
                )
                net.load_parameters(best_epoch_info.params_path, self.ctx)

                logger.info(
                    f"Final loss: {best_epoch_info.metric_value} "
                    f"(occurred at epoch {best_epoch_info.epoch_no})"
                )

                # save net parameters
                net.save_parameters(best_epoch_info.params_path)

                logger.info("End model training")

                sw.close()
