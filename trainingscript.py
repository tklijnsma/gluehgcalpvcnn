import os, os.path as osp, math, numpy as np, tqdm, logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (
    NNConv, graclus, max_pool, max_pool_x,
    global_mean_pool
    )

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Dir of this file
GLUEDIR = osp.abspath(osp.dirname(__file__))

import utils
logger = logging.getLogger('root')


class LindseysTrainingScript(object):
    """docstring for LindseysTrainingScript"""
    def __init__(self):
        super(LindseysTrainingScript, self).__init__()

        self.directed = False
        self.train_batch_size = 1
        self.valid_batch_size = 1
        self.n_epochs = 50

        self.dataset_path = osp.join(
            GLUEDIR,
            'data/single-muon-july2019-npz'
            )

        # self.categorized = False
        # self.forcecats = False
        self.categorized = True
        self.forcecats = True
        self.cats = 1

        # self.model_name = 'EdgeNet2'
        self.model_name = 'EdgeNetWithCategories'
        self.loss = 'binary_cross_entropy'
        self.optimizer = 'Adam'
        self.hidden_dim = 64
        self.n_iters = 6
        self.lr = 0.01

        self.output_dir = osp.join(GLUEDIR, '../output')


    def run_edgenet(self):
        from datasets.hitgraphs import HitGraphDataset
        from training.gnn import GNNTrainer

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info('using device %s'%device)

        logger.info(self.dataset_path)
        full_dataset = HitGraphDataset(
            self.dataset_path,
            directed = self.directed,
            categorical = self.categorized
            )

        fulllen = len(full_dataset)
        tv_frac = 0.20
        tv_num = math.ceil(fulllen*tv_frac)
        splits = np.cumsum([fulllen-tv_num,0,tv_num])
        logger.info('%s, %s', fulllen, splits)

        train_dataset = torch.utils.data.Subset(
            full_dataset,
            # np.arange(start=0,stop=splits[1])
            # list(range(0, splits[0]))
            list(range(0, 50)) # For quicker debugging
            )
        valid_dataset = torch.utils.data.Subset(
            full_dataset,
            # np.arange(start=splits[1],stop=splits[2])
            # list(range(splits[1], splits[2]))
            list(range(50, 60)) # For quicker debugging
            )
        train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.valid_batch_size, shuffle=False)

        train_samples = len(train_dataset)
        valid_samples = len(valid_dataset)

        d = full_dataset
        num_features = d.num_features
        num_classes = d[0].y.dim() if d[0].y.dim() == 1 else d[0].y.size(1)
        
        if self.categorized:
            if not self.forcecats:
                num_classes = int(d[0].y.max().item()) + 1 if d[0].y.dim() == 1 else d[0].y.size(1)
            else:
                num_classes = self.cats
        logger.debug('num_classes = %s', num_classes)

        # the_weights = np.array([1., 1., 1., 1.]) #[0.017, 1., 1., 10.]
        the_weights = np.array([1.]) # Only signal and noise now
        trainer = GNNTrainer(
            category_weights = the_weights, 
            output_dir = self.output_dir,
            device = device
            )
        trainer.logger.setLevel(logging.DEBUG)
        trainer.logger.addHandler(logger.handlers[0]) # Just give same handler as other log messages

        #example lr scheduling definition
        def lr_scaling(optimizer):
            from torch.optim.lr_scheduler import ReduceLROnPlateau        
            return ReduceLROnPlateau(
                optimizer, mode='min', verbose=True,
                min_lr=5e-7, factor=0.2, 
                threshold=0.01, patience=5
                )
        
        trainer.build_model(
            name          = self.model_name,
            loss_func     = self.loss,
            optimizer     = self.optimizer,
            learning_rate = self.lr,
            lr_scaling    = lr_scaling,
            input_dim     = num_features,
            hidden_dim    = self.hidden_dim,
            n_iters       = self.n_iters,
            output_dim    = num_classes
            )
        
        trainer.print_model_summary()

        train_summary = trainer.train(train_loader, self.n_epochs, valid_data_loader=valid_loader)
        logger.info(train_summary)
