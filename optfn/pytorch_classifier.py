import copy

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from .feedforward_networks import FCNet
from .virtual_adversarial_training import get_vat_loss


def optimizer_factory(params):
    weights, biases = [], []
    for name, p in params:
       if 'weight' in name:
           weights.append(p)
       else:
           biases.append(p)
    params = [
      dict(params=weights),
      dict(params=biases, weight_decay=0)
    ]
    return torch.optim.Adam(params, lr=1e-4, weight_decay=1e-4)


class PytorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, use_best_model=True,
                 model_factory=lambda i, o: FCNet(i, o), early_stopping_epochs=20,
                 optimizer_factory=optimizer_factory, vat_eps=None, vat_batch_size=256,
                 lr_scheduler_factory=lambda opt: MultiStepLR(opt, milestones=[20, 40, 60, 80], gamma=0.25),
                 train_epochs=50, batch_size=256, enable_grad_noise=False, grad_noise=(0.1, 0.55), log_interval=None):
        self.model_factory = model_factory
        self.optimizer_factory = optimizer_factory
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.lr_scheduler_factory = lr_scheduler_factory
        self.grad_noise = grad_noise if enable_grad_noise else None
        self.log_interval = log_interval
        self.early_stopping_epochs = early_stopping_epochs
        self.return_best = use_best_model
        self.vat_eps = vat_eps
        self.vat_batch_size = vat_batch_size
        self.label_encoder = None
        self._model = None
        self._optimizer = None
        self._lr_scheduler = None
        self.classes_ = None
        self.x_, self.y_ = None, None
        self._cur_grad_noise_std = 0

    def fit(self, x_train, y_train, x_val=None, y_val=None, x_us=None):
        assert (x_val is None) == (y_val is None)

        x_train, y_train = check_X_y(x_train, y_train)
        if x_val is not None:
            x_val, y_val = check_X_y(x_val, y_val)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y_train.ravel())
        self.classes_ = self.label_encoder.classes_

        self.x_ = x_train
        self.y_ = y_train

        if x_us is None and self.vat_eps is not None:
            x_us = np.concatenate((x_train, x_val), 0)
        x_train, y_train = self._preprocess_xy(x_train, y_train)
        if x_val is not None:
            x_val, y_val = self._preprocess_xy(x_val, y_val)
        if x_us is not None:
            x_us, _ = self._preprocess_xy(x_us, None)

        self._model = self.model_factory(x_train.size(1), len(self.classes_))
        self._optimizer = self.optimizer_factory(self._model.named_parameters())
        if self.lr_scheduler_factory is not None:
            self._lr_scheduler = self.lr_scheduler_factory(self._optimizer)
        train_loader = self._make_dataloader(x_train, y_train, True)

        val_loader = self._make_dataloader(x_val, y_val, False) if x_val is not None else None
        us_loader = self._make_dataloader(x_us, None, True, batch_size=self.vat_batch_size) if x_us is not None else None
        best_state_dict = None
        min_val_loss = float('inf')
        last_improved_epoch = 0

        for i in range(self.train_epochs):
            self._train_epoch(i, train_loader, us_loader)
            if x_val is not None:
                log = self.log_interval is not None and i % self.log_interval == 0
                val_loss = self._calc_loss(val_loader)
                if self.return_best and val_loss < min_val_loss:
                    min_val_loss = val_loss
                    best_state_dict = copy.deepcopy(self._model.state_dict())
                    last_improved_epoch = i
                    if log:
                        print(i, 'new best val loss:', val_loss)
                elif log:
                    print(i, 'val loss:', val_loss)
                if self.log_interval is not None and i - last_improved_epoch > self.early_stopping_epochs:
                    print('early stopping at epoch', i)
                    break

        if self.log_interval is not None and self.return_best and x_val is not None:
            print('loading state for val loss', min_val_loss)
            self._model.load_state_dict(best_state_dict)

        return self

    def predict_proba(self, x, return_numpy=True):
        check_is_fitted(self, ['x_', 'y_'])
        x = check_array(x)

        if return_numpy:
            x, _ = self._preprocess_xy(x, None)

        loader = self._make_dataloader(x, None, False)
        outputs = []

        self._model.eval()
        for batch_idx, data in enumerate(loader):
            if self._cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            pred = self._model(data).data
            outputs.append(pred.cpu().numpy() if return_numpy else pred)

        if return_numpy:
            return np.exp(np.vstack(outputs))
        else:
            return torch.cat(outputs, 0)

    def _calc_loss(self, loader):
        total_loss = 0
        total_samples = 0
        self._model.eval()
        for batch_idx, (data, target) in enumerate(loader):
            if self._cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            pred = self._model(data)
            loss = F.nll_loss(pred, target)
            total_loss += loss.data[0]*data.size(0)
            total_samples += data.size(0)
        return total_loss/total_samples

    def _preprocess_xy(self, x, y):
        x = np.atleast_2d(x)
        assert x.ndim == 2, x.ndim
        x = x.astype(np.float32, copy=False)
        x = torch.from_numpy(x)

        if y is not None:
            y = np.atleast_1d(y)
            if y.ndim != 1:
                y = y.ravel()
            assert y.ndim == 1, y.ndim
            assert x.size(0) == y.shape[0], (x.size(), y.shape)
            y = self.label_encoder.transform(y)
            y = y.astype(np.int64, copy=False)
            y = torch.from_numpy(y)

        return x, y

    def _make_dataloader(self, x, y, shuffle, batch_size=None):
        dataset = torch.utils.data.TensorDataset(x, y) if y is not None else MultiDataset(x)
        kwargs = {'num_workers': 1, 'pin_memory': True} if self._cuda else {}
        batch_size = self.batch_size if batch_size is None else batch_size
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return loader

    @property
    def _cuda(self):
        return next(self._model.parameters()).is_cuda

    def _train_epoch(self, epoch_idx, loader, us_loader):
        self._model._train()
        total_loss = 0
        total_samples = 0
        if us_loader is not None:
            us_enum = enumerate(us_loader)
        for batch_idx, (data, target) in enumerate(loader):
            if self._cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self._optimizer.zero_grad()
            output = self._model(data)
            loss = F.nll_loss(output, target)
            loss_bwd = loss
            if self.vat_eps is not None:
                us_data = next(us_enum, None)
                if us_data is None:
                    us_enum = enumerate(us_loader)
                    us_data = next(us_enum)
                us_data = Variable(us_data[1].type_as(data.data))
                loss_bwd += get_vat_loss(self._model, us_data, eps=self.vat_eps)
            if hasattr(self._model, 'extra_loss'):
                loss_bwd += loss + self._model.extra_loss()
            loss_bwd.backward()
            if self.grad_noise is not None:
                self._cur_grad_noise_std = np.sqrt(self.grad_noise[0] / np.power(1 + epoch_idx, self.grad_noise[1]))
                self._model.apply(self._add_grad_noise)

            # clip_grad_norm(self._model.parameters(), 10)

            self._optimizer.step()
            if self._lr_scheduler is not None:
                self._lr_scheduler.step(loss.data[0])
            total_loss += loss.data[0]*data.size(0)
            total_samples += data.size(0)
        if self.log_interval is not None and epoch_idx % self.log_interval == 0:
            print(epoch_idx, 'train loss:', total_loss/total_samples)

    def _add_grad_noise(self, param):
        if hasattr(param, 'grad') and param.grad is not None:
            param.grad += torch.randn(param.size())*self._cur_grad_noise_std


