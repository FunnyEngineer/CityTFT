from dataset.dataset import CitySimDataModule
import lightning as L
from model.rnn import RNNSeqNetV2
from model.transformer import TransNetV2
import pdb
from sklearn.metrics import f1_score
import torch
import pickle

from model.model_tft import TemporalFusionTransformer
from configuration import CONFIGS

input_dim = 26
input_seq_len = 24

cool_dict = {'RNN': '#69A2B0', 'Transformer': '#659157'}  # '#F4D35E', '#EE964B', '#F95738'


def cal_metric(pred_dict):
    for mode in pred_dict:
        predictions = pred_dict[mode]
        # heat_pred = torch.tensor([])
        # cool_pred = torch.tensor([])
        # heat_true = torch.tensor([])
        # cool_true = torch.tensor([])
        # for batch in predictions:
        #     heat_true = torch.cat([heat_true, batch[1][:, :, 0].eq(0).float()])
        #     heat_pred = torch.cat([heat_pred, batch[0][0].eq(0).float()])
        #     cool_true = torch.cat([cool_true, batch[1][:, :, 1].eq(0).float()])
        #     cool_pred = torch.cat([cool_pred, batch[0][1].eq(0).float()])

        # # calculate F1 score
        # heat_f1 = f1_score(heat_true.cpu().numpy().flatten(), heat_pred.cpu().numpy().flatten())
        # cool_f1 = f1_score(cool_true.cpu().numpy().flatten(), cool_pred.cpu().numpy().flatten())
        # total_f1 = f1_score(torch.cat([heat_true, cool_true]).cpu().numpy(
        # ).flatten(), torch.cat([heat_pred, cool_pred]).cpu().numpy().flatten())
        # print(f'{mode} F1 score: {total_f1:.4f} (heat: {heat_f1:.4f}, cool: {cool_f1:.4f})')

        heat_pred = torch.tensor([])
        cool_pred = torch.tensor([])
        heat_true = torch.tensor([])
        cool_true = torch.tensor([])
        for batch in predictions:
            heat_true = torch.cat([heat_true, batch[1][:, :, 0].flatten()])
            heat_pred = torch.cat([heat_pred, batch[0][0].flatten()])
            cool_true = torch.cat([cool_true, batch[1][:, :, 1].flatten()])
            cool_pred = torch.cat([cool_pred, batch[0][1].flatten()])

        # calculate non-zero MAPE
        heat_mape = torch.mean(
            torch.abs((heat_true[heat_true != 0]-heat_pred[heat_true != 0])/heat_true[heat_true != 0]))
        cool_mape = torch.mean(
            torch.abs((cool_true[cool_true != 0]-cool_pred[cool_true != 0])/cool_true[cool_true != 0]))
        total_mape = torch.mean(torch.abs((torch.cat([heat_true[heat_true != 0], cool_true[cool_true != 0]])-torch.cat(
            [heat_pred[heat_true != 0], cool_pred[cool_true != 0]]))/(torch.cat([heat_true[heat_true != 0], cool_true[cool_true != 0]]))))
        print(f'{mode} MAPE: {total_mape:.4f} (heat: {heat_mape:.4f}, cool: {cool_mape:.4f})')
        # calculate overall RMSE
        # heat_rmse = torch.sqrt(torch.mean((heat_true-heat_pred)**2))
        # cool_rmse = torch.sqrt(torch.mean((cool_true-cool_pred)**2))
        # total_rmse = torch.sqrt(torch.mean(
        #     (torch.cat([heat_true, cool_true])-torch.cat([heat_pred, cool_pred]))**2))
        # print(f'{mode} RMSE: {total_rmse:.4f} (heat: {heat_rmse:.4f}, cool: {cool_rmse:.4f})')

        # # cauculate only non-zero RMSE
        # heat_rmse = torch.sqrt(torch.mean((heat_true[heat_true != 0]-heat_pred[heat_true != 0])**2))
        # cool_rmse = torch.sqrt(torch.mean((cool_true[cool_true != 0]-cool_pred[cool_true != 0])**2))
        # total_rmse = torch.sqrt(torch.mean(
        #     (torch.cat([heat_true[heat_true != 0], cool_true[cool_true != 0]])-torch.cat([heat_pred[heat_true != 0], cool_pred[cool_true != 0]]))**2))
        # print(f'{mode} RMSE (non-zero): {total_rmse:.4f} (heat: {heat_rmse:.4f}, cool: {cool_rmse:.4f})')


def visual_load_differences():

    # load the model
    # rnn = RNNSeqNetV2(input_dim=input_dim, input_ts=input_seq_len, output_ts=input_seq_len).load_from_checkpoint(
    #     'multi_cli/rnn_with_prob_quantile_v3_adamW_lr1e-4/checkpoints/epoch=00-step=9103.0546875-val_loss=0.85979670-v2.ckpt') # best
    #     # 'multi_cli/rnn_with_prob_quantile_v3_adamW_lr1e-4/checkpoints/epoch=398-step=3632125.25-val_loss=2.11352944.ckpt')  # last
    # trans = TransNetV2(input_dim=input_dim, input_ts=input_seq_len, output_ts=input_seq_len).load_from_checkpoint(
    #     'multi_cli/trans_v3_with_prob_quantile_adamW_lr1e-4/checkpoints/epoch=01-step=18206.109375-val_loss=0.81011742-v1.ckpt') # best
    #     # 'multi_cli/trans_v3_with_prob_quantile_adamW_lr1e-4/checkpoints/last.ckpt')  # last
    # trans.eval()

    # config = CONFIGS['citysim']()
    # checkpoint = torch.load('multi_cli/tft_with_prob_v2_adamw_lr1e-4/checkpoints/last.ckpt')
    # tft = TemporalFusionTransformer(config) # last
    # tft.load_state_dict(checkpoint['state_dict'])
    # # 'multi_cli/tft_with_prob_v2_adamw_lr1e-4/checkpoints/epoch=53-step=491569.46875-val_loss=0.00555553-v1.ckpt') # best
    # tft.eval()
    # dm = CitySimDataModule(input_ts=input_seq_len, output_ts=input_seq_len, mode='tft')
    # dm.setup(stage='test')

    # test the model
    # trainer = L.Trainer()
    # trans_pred = trainer.predict(trans, dm.test_dataloader())
    # rnn_pred = trainer.predict(rnn, dm.test_dataloader())
    # tft_pred = trainer.predict(tft, dm.test_dataloader())

    # store the predictions
    # pickle.dump(rnn_pred, open('predictions/rnn_best.pkl', 'wb'))
    # pickle.dump(trans_pred, open('predictions/trans_best.pkl', 'wb'))
    # pickle.dump(tft_pred, open('predictions/tft_last.pkl', 'wb'))

    rnn_pred = pickle.load(open('predictions/rnn_last.pkl', 'rb'))
    trans_pred = pickle.load(open('predictions/trans_last.pkl', 'rb'))
    tft_pred = pickle.load(open('predictions/tft_last.pkl', 'rb'))

    # cal_metric({'RNN': rnn_pred, 'Transformer': trans_pred})
    # pdb.set_trace()
    cal_metric({'RNN': rnn_pred, 'Transformer': trans_pred, 'TFT': tft_pred})


if __name__ == '__main__':
    visual_load_differences()
