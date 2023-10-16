import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.utils.data as data
from collections import defaultdict


from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from lightning import LightningModule
from lightning.pytorch import Trainer, LightningDataModule


from load_datasets import dataset_dict
from losses import loss_dict
from models.nerf import *
from utils.utils import *
from models.rendering import *
import opt


class NeRFLightningModule(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.example_input_array = torch.Tensor(640000, 8)
        self.val_outputs = {'val_loss': [], 'val_psnr': []}


        self.loss = loss_dict['color'](coef=1)

        self.embedding_xyz = Embedding(hparams.N_emb_xyz)
        self.embedding_dir = Embedding(hparams.N_emb_dir)
        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}

        self.nerf_coarse = NeRF(in_channels_xyz=6 * hparams.N_emb_xyz + 3,
                                in_channels_dir=6 * hparams.N_emb_dir + 3)
        self.models = {'coarse': self.nerf_coarse}
        load_ckpt(self.nerf_coarse, hparams.weight_path, 'nerf_coarse')

        if hparams.N_importance > 0:
            self.nerf_fine = NeRF(in_channels_xyz=6 * hparams.N_emb_xyz + 3,
                                  in_channels_dir=6 * hparams.N_emb_dir + 3)
            self.models['fine'] = self.nerf_fine
            load_ckpt(self.nerf_fine, hparams.weight_path, 'nerf_fine')

    def forward(self, rays):
        num_rays = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, num_rays, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i + self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk,  # chunk size is effective in val mode
                            self.train_dataset.white_back)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def setup(self, stage: str):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh)}
        if self.hparams.dataset_name == 'llff':
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = self.hparams.num_gpus
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1,  # validate one image (H*W rays) at a time
                          pin_memory=True)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [scheduler]

    def training_step(self, batch, batch_nb):
        rays, rgbs = batch['rays'], batch['rgbs']
        results = self(rays)
        loss = self.loss(results, rgbs)

        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)

        log = {'lr': get_learning_rate(self.optimizer), 'train/loss': loss, 'train/psnr': psnr_}
        self.log_dict(log, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):
        rays, rgbs = batch['rays'], batch['rgbs']
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        results = self(rays)
        loss = self.loss(results, rgbs)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        # 在第一个批次时，可视化模型的输出，包括预测的图像 (img)、GT 图像 (img_gt) 和深度图 (depth)。
        if batch_nb == 0:
            W, H = self.hparams.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W))  # (3, H, W)
            stack = torch.stack([img_gt, img, depth])  # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                              stack, self.global_step)

        psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        log = {'val_loss': loss, 'val_psnr': psnr_}
        self.log_dict(log, on_epoch=True, prog_bar=True)
        self.val_outputs['val_loss'].append(loss)
        self.val_outputs['val_psnr'].append(psnr_)
        return loss

    def on_validation_epoch_end(self):
        # mean_loss = torch.stack([x for x in self.val_outputs['val_loss']]).mean()
        # mean_psnr = torch.stack([x for x in self.val_outputs['val_psnr']]).mean()
        mean_loss = torch.mean(torch.tensor(self.val_outputs['val_loss']))
        mean_psnr = torch.mean(torch.tensor(self.val_outputs['val_psnr']))
        for key in self.val_outputs:
            self.val_outputs[key] = []  # 清空每个键对应的列表，将其设置为空列表

        log = {'val/loss': mean_loss, 'val/psnr': mean_psnr}
        # self.log('val/loss', mean_loss)
        # self.log('val/psnr', mean_psnr, prog_bar=True)
        self.log_dict(log, prog_bar=True)


def main(hparams):
    # nerf_data = NeRFDataModule(hparams=hparams)
    nerf_pl = NeRFLightningModule(hparams)
    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}',
                              filename='model-{epoch:02d}-{val/psnr:.2f}',
                              monitor='val/psnr',
                              mode='max',
                              save_top_k=1)
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [ModelSummary(max_depth=-1), ckpt_cb, pbar]

    logger = TensorBoardLogger(save_dir="logs",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=callbacks,
                      # resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='auto',
                      devices=hparams.num_gpus,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler="simple" if hparams.num_gpus == 1 else None)

    # trainer.fit(nerf_pl, datamodule=nerf_data, ckpt_path=hparams.ckpt_path)
    trainer.fit(nerf_pl, ckpt_path=hparams.ckpt_path)
    # trainer.fit(nerf_pl)



if __name__ == '__main__':
    hparams = opt.get_opts()
    main(hparams)
