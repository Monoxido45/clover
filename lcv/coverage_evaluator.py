from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from optuna.trial import TrialState
from sklearn.base import BaseEstimator
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.optim import Adamax as optimm
from torch.utils import data


class Coverage_evaluator(BaseEstimator):
    """
    Neural Network Coverage Evaluator used as intermediary model in conditional coverage
    hypothesis testing. The learning rate, optimizers, number of hidden layers,
    number of hidden states and dropout ratios hyperparameters are all optimized.
    -----------
    seed: integer
        Controls the randommness of neural network modelling components such as batches, dropouts and optimization techniques
    early_stopping: integer
        Number of epochs with no model improvement after which training will be stopped.
    nepoch: integer
        Number of epochs to run
    batch_size: integer
        Training batch size
    batch_test_size: integer
        Validation batch_size
    validation_split: float
        Proportion of data used for validation
    nlayers: positive integer
        Number of layers. Default is 3.
    hidden_size: Positive Integer or List of integers or None
        List of hidden unit  with length equal to nlayers. Default is 100 for each layer.
    splitter_seed: int
        Control the randomness of training/validation splitting
    gpu: bool
        If true, will use gpu for computation, if available
    scale_data: bool
        If true, will perform feature scaling before fitting
    """

    def __init__(
        self,
        seed=1250,
        early_stopping=20,
        nepoch=100,
        batch_size=150,
        batch_test_size=1000,
        validation_split=0.33,
        dataloader_workers=1,
        splitter_seed=1250,
        nlayers=3,
        hidden_size=100,
        gpu=True,
        scale_data=True,
        optimized=False,
        verbose_optim=0,
    ):
        for prop in dir():
            if prop != "self":
                setattr(self, prop, locals()[prop])
        self.gpu = gpu

    def move_to_gpu(self):
        self.model.cuda()
        self.gpu = True

        return self

    def move_to_cpu(self):
        self.model.cpu()
        self.gpu = False

        return self

    # optimizing and fitting neural net
    def fit(self, X_train, y_train):
        # transforming X and y into tensors
        self.gpu = self.gpu and torch.cuda.is_available()
        self.x_dim = X_train.shape[1]
        self.x_train = X_train
        self.y_train = y_train

        if self.scale_data:
            self.scaler = StandardScaler()
            self.scaler.fit(self.x_train)
            self.x_train = self.scaler.transform(self.x_train)

        if self.gpu:
            DEVICE = torch.device("cuda")
        else:
            DEVICE = torch.device("cpu")

        # obtaining optimized nnet
        if not self.optimized:
            best_params = self.optimize()
            self.optimized = True
            # optimized neural net constructor
            torch.manual_seed(self.seed)
            self.model = self._build_model(best_params, optimize=False).to(DEVICE)
            # xavier normal initialization
            self.model.apply(self.reset_weights)

            # best optimizer and learning rate
            opt = getattr(optim, best_params["optimizer"])
            lr = best_params["lr"]
            self.optimizer = opt(self.model.parameters(), lr=lr)
        else:
            # resetting weights using xavier normal
            self.model.apply(self.reset_weights)

        return self.improve_fit(self.optimizer, self.nepoch)

    def improve_fit(self, optimizer, nepoch):
        # loss function of interest
        loss_function = nn.BCELoss()

        # obtaining train and validation data
        splitter = ShuffleSplit(
            n_splits=1, test_size=self.validation_split, random_state=self.splitter_seed
        )
        index_train, index_val = next(iter(splitter.split(self.x_train, self.y_train)))
        self.index_train = index_train
        self.index_val = index_val

        inputv_train = np.array(self.x_train, dtype="f4")
        target_train = np.array(self.y_train, dtype=np.int64)

        # inputs for validation
        inputv_val = inputv_train[index_val]
        target_val = target_train[index_val]
        inputv_val = np.ascontiguousarray(inputv_val)
        target_val = np.ascontiguousarray(target_val)

        # inputs for training
        inputv_train = inputv_train[index_train]
        target_train = target_train[index_train]
        inputv_train = np.ascontiguousarray(inputv_train)
        target_train = np.ascontiguousarray(target_train)

        # initializing best loss and loss list
        self.best_val_loss = np.infty
        self.loss_history_validation = []
        self.loss_history_train = []
        self.epoch_list = []

        # for early stopping
        es_tries = 0

        # training and validating model
        for epoch in range(nepoch):
            # first training
            self.model.train()
            trainloss = self._one_epoch(
                True,
                self.batch_size,
                inputv_train,
                target_train,
                optimizer,
                loss_function,
            )
            self.loss_history_train.append(trainloss)

            # Then validating
            self.model.eval()
            avloss = self._one_epoch(
                False,
                self.batch_test_size,
                inputv_val,
                target_val,
                optimizer,
                loss_function,
            )

            self.loss_history_validation.append(avloss)
            self.epoch_list.append(epoch)

            if avloss <= self.best_val_loss:
                self.best_val_loss = avloss
                es_tries = 0
                best_state_dict = self.model.state_dict()
                best_state_dict = deepcopy(best_state_dict)
                self.best_loss_history_validation = avloss
            else:
                es_tries += 1

            if es_tries == self.early_stopping:
                self.model.load_state_dict(best_state_dict)
                return self

        return self

    def plot_history(self):
        plt.plot(self.epoch_list, self.loss_history_train, color="tab:red")
        plt.plot(self.epoch_list, self.loss_history_validation, color="tab:blue")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

    def return_history_dict(self):
        return {
            "epoch": self.epoch_list,
            "train loss": self.loss_history_train,
            "valid loss": self.loss_history_validation,
        }

    # optimizing by optuna study
    def optimize(self):
        # adjusting optuna verbosity level
        if self.verbose_optim == 0:
            optuna.looging.set_verbosity(optuna.logging.WARNING)
        else:
            optuna.looging.set_verbosity(optuna.logging.INFO)
        # creating study object
        study = optuna.create_study(
            sampler=optuna.samplers.RandomSampler(seed=self.seed), direction="minimize"
        )
        # optimizing
        study.optimize(self._objective, n_trials=150, timeout=600)

        # completing all trials
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        best_params = study.best_params
        return best_params

    # optuna based function to build nnet
    def _build_model(self, trial, optimize=True):
        # initiating model to optimize
        # optimizing number of layers
        if optimize:
            n_layers = self.nlayers
            layers = []
            in_features = self.x_dim

            for i in range(n_layers):
                # optimizing number of hidden units
                out_features = self.hidden_size
                layers.append(nn.Linear(in_features, out_features))
                layers.append(nn.SELU())
                # optimizing dropout ratio
                p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.7)
                layers.append(nn.Dropout(p))
                in_features = out_features

        # if already optimized, we set trial as param_dict
        else:
            param_dict = trial
            layers = []
            in_features = self.x_dim
            for i in range(n_layers):
                # number of hidden units for i-th layer
                out_features = self.hidden_size
                layers.append(nn.Linear(in_features, out_features))
                layers.append(nn.SELU())
                p = param_dict["dropout_l{}".format(i)]
                layers.append(nn.Dropout(p))
                in_features = out_features

        layers.append(nn.Linear(in_features, 1))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    # fitting neural net for optuna
    def _objective(self, trial):
        if self.gpu:
            DEVICE = torch.device("cuda")
        else:
            DEVICE = torch.device("cpu")

        torch.manual_seed(self.seed)
        # Generate the model
        self.model = self._build_model(trial).to(DEVICE)

        # trying several values of learning rate
        lr = trial.suggest_float("lr", 1e-5, 0.25, log=True)
        optimizer = optimm(self.model.parameters(), lr=lr)

        # loss function of interest
        loss_function = nn.BCELoss()

        # obtaining train and validation data
        splitter = ShuffleSplit(
            n_splits=1, test_size=self.validation_split, random_state=self.splitter_seed
        )
        index_train, index_val = next(iter(splitter.split(self.x_train, self.y_train)))
        self.index_train = index_train
        self.index_val = index_val

        inputv_train = np.array(self.x_train, dtype="f4")
        target_train = np.array(self.y_train, dtype=np.int64)

        # inputs for validation
        inputv_val = inputv_train[index_val]
        target_val = target_train[index_val]
        inputv_val = np.ascontiguousarray(inputv_val)
        target_val = np.ascontiguousarray(target_val)

        # inputs for training
        inputv_train = inputv_train[index_train]
        target_train = target_train[index_train]
        inputv_train = np.ascontiguousarray(inputv_train)
        target_train = np.ascontiguousarray(target_train)

        # initializing best loss and loss list
        self.best_val_loss = np.infty
        self.loss_history_validation = []

        # for early stopping
        es_tries = 0

        # training and validating model
        for epoch in range(self.nepoch):
            # first training
            self.model.train()
            self._one_epoch(
                True,
                self.batch_size,
                inputv_train,
                target_train,
                optimizer,
                loss_function,
            )

            # Then validating
            self.model.eval()
            avloss = self._one_epoch(
                False,
                self.batch_test_size,
                inputv_val,
                target_val,
                optimizer,
                loss_function,
            )
            self.loss_history_validation.append(avloss)

            if avloss <= self.best_val_loss:
                self.best_val_loss = avloss
                es_tries = 0
                self.best_loss_history_validation = avloss
            else:
                es_tries += 1

            trial.report(avloss, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if es_tries == self.early_stopping:
                return self.best_loss_history_validation

        return self.best_loss_history_validation

    def reset_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.constant_(m.bias, 0)
            gain = nn.init.calculate_gain("relu")
            nn.init.xavier_normal_(m.weight, gain=gain)

    # one epoch fitting
    def _one_epoch(self, is_train, batch_size, inputv, target, optimizer, criterion):
        with torch.set_grad_enabled(is_train):
            inputv = torch.from_numpy(inputv)
            target = torch.from_numpy(target)
            loss_vals = []

            tdataset = data.TensorDataset(inputv, target)
            data_loader = data.DataLoader(
                tdataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=is_train,
                pin_memory=self.gpu,
                num_workers=self.dataloader_workers,
            )

            for inputv_this, target_this in data_loader:
                if self.gpu:
                    inputv_this = inputv_this.cuda(non_blocking=True)
                    target_this = target_this.cuda(non_blocking=True)
                else:
                    inputv_this = inputv_this.cpu(non_blocking=True)
                    target_this = target_this.cpu(non_blocking=True)

                inputv_this.requires_grad_(True)

                optimizer.zero_grad()
                output = self.model(inputv_this)

                loss = criterion(torch.reshape(output, (-1,)), target_this.float())
                np_loss = loss.data.item()
                loss_vals.append(np_loss)

                if is_train:
                    loss.backward()
                    optimizer.step()

            avgloss = np.average(loss_vals)

            return avgloss

    def predict(self, x_pred):
        if self.scale_data:
            x_pred = self.scaler.transform(x_pred)

        with torch.no_grad():
            self.model.eval()
            inputv = np.array(x_pred, dtype="f4")
            inputv = torch.from_numpy(inputv)

            if self.gpu:
                inputv = inputv.cuda()

            pred = self.model(inputv)

        output_pred = pred.data.cpu().numpy()
        return output_pred
