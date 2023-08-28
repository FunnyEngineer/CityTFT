from model import Net
from dataset import CitySimDataModule
import lightning as L
import torch
torch.set_float32_matmul_precision('highest')


model = Net().load_from_checkpoint('lightning_logs/version_3/checkpoints/epoch=594-step=5942265.ckpt')


trainer = L.Trainer()
dm = CitySimDataModule()
dm.prepare_data()
dm.setup(stage='test')
trainer.test(model, datamodule=dm)
# trainer.predict(model, datamodule=dm)
