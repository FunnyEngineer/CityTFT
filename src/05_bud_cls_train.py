from dataset.building_dataset import BuildingClsDataset, BuildingClsModule
from model.model_building import BudNet

from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import torch
torch.set_float32_matmul_precision('highest')

def setting_logger():
    # store the best model and the last model
    save_best = ModelCheckpoint(
        save_top_k=3,
        monitor='val/total_loss',
        mode='min',
        save_last=True,
        filename='epoch={epoch:02d}-step={global_step}-val_loss={val/total_loss:.8f}',
        auto_insert_metric_name=False,
    )
    save_last = ModelCheckpoint(
        save_top_k=3,
        monitor="global_step",
        mode="max",
        filename='epoch={epoch:02d}-step={global_step}-val_loss={val/total_loss:.8f}',
        auto_insert_metric_name=False,
    )
    return save_best, save_last


def bud_train():
    logger = TensorBoardLogger('bud_cls', version='dnn_v0')

    save_best, save_last = setting_logger()

    # train the model
    trainer = L.Trainer(max_epochs=400, logger=logger, check_val_every_n_epoch=1,
                        callbacks=[save_last, save_best])
    
    # init datamodule
    dm = BuildingClsModule()
    dm.setup(stage='fit')

    # init model
    model = BudNet()

    # train the model
    trainer.fit(model, datamodule=dm)
    dm.setup(stage='test')

    # test the model
    trainer.test(model, datamodule=dm)

if __name__ == '__main__':
    L.seed_everything(1340)
    bud_train()