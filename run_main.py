import argparse
import os
import torch
from utils.print_args import print_args
import random
import numpy as np

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy, vali, del_files
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb
from models import GPT2LMTS, GPTJLMTS
from tqdm import tqdm
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
import json
from utils.analysis import Analyzer

from torch.optim import lr_scheduler
if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # add for seed
    os.environ['PYTHONHASHSEED'] = str(fix_seed)
    torch.cuda.manual_seed(fix_seed)
    torch.cuda.manual_seed_all(fix_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')
    parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')


    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # classification gpt embed
    parser.add_argument('--kernel_width', type=int, default=3)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--padding', type=int, default=1)
    parser.add_argument('--llm_dim', type=int, default='768', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)


    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=0,
                        help='1: channel dependence 0: channel independence for FreTS model')
    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    args = parser.parse_args()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)
    from accelerate.logging import get_logger

    logger = get_logger(__name__, log_level="DEBUG")
    # log all processes
    logger.debug("thing_to_log", main_process_only=False)

    if args.is_training:
        for ii in range(args.itr):
            setting = '{}_{}_{}_{}_kw{}_st{}_pa{}_llm{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.kernel_width,
                args.stride,
                args.padding,
                args.llm_dim,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, 
                ii)

        train_data, train_loader = data_provider(args, 'TRAIN')
        #vali_data, vali_loader = data_provider(args, 'TEST')
        test_data, test_loader = data_provider(args, 'TEST')
        
        # Only for GPT2, GPTJ
        if args.model == 'GPT2':
            model = GPT2LMTS.Model(args).float()
        elif args.model == 'GPTJ':
            model = GPTJLMTS.Model(args).float()


        path = os.path.join(args.checkpoints,
                            setting + '-' + args.model_comment)  # unique checkpoint saving path
        
        if not os.path.exists(path) and accelerator.is_local_main_process:
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience, save_mode=False)

        trained_parameters = []
        for p in model.parameters():
            if p.requires_grad is True:
                trained_parameters.append(p)

        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

        if args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=0.2,
                                                epochs=args.train_epochs,
                                                max_lr=args.learning_rate)

        criterion = nn.CrossEntropyLoss()
        #train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        #    train_loader, vali_loader, test_loader, model, model_optim, scheduler)
        train_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
            train_loader, test_loader, model, model_optim, scheduler)

        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        analyzer = Analyzer(print_conf_mat=True)
        best_acc = 0.0
        completed_steps = 0
        for epoch in range(args.train_epochs):
            iter_count = 0
            train_loss = []

            model.train()
            epoch_time = time.time()
            print(model_optim.param_groups[0]['lr'])
            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                with accelerator.accumulate(model):
                    iter_count += 1

                    batch_x = batch_x.float().to(accelerator.device)
                    padding_mask = padding_mask.float().to(accelerator.device)
                    label = label.to(accelerator.device)
                    
                    outputs = model(batch_x, padding_mask, None, None)
                    loss = criterion(outputs, label.long().squeeze(-1))
                    train_loss.append(loss.item())

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 4.0)

                    model_optim.step()
                    model_optim.zero_grad()


                # if args.lradj == 'TST':
                #     adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                #     scheduler.step()

            accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(train_loss)
            
            #vali_loss, val_accuracy = vali(args, accelerator, model, vali_data, vali_loader, criterion, analyzer, "VAL")
            test_loss, test_accuracy = vali(args, accelerator, model, test_data, test_loader, criterion, analyzer, "TEST")
            vali_loss = test_loss
            val_accuracy = test_accuracy
            accelerator.print(
                "Epoch: {0}| Train Loss: {1:.3f} Vali Loss: {2:.3f} Vali Acc: {3:.3f} Test Loss: {4:.3f} Test Acc: {5:.3f}"
                .format(epoch + 1, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            early_stopping(-val_accuracy, model, path)
            model_optim.zero_grad()
            if test_accuracy > best_acc:
                best_acc = test_accuracy
                if not os.path.exists(path) and accelerator.is_local_main_process:
                    os.makedirs(path)
                accelerator.print("Save Best Results")
                other_path = str(args.learning_rate) + "_" + str(epoch)+"_all_results.json"
                with open(os.path.join(path, other_path), "w") as f:
                    json.dump({
                    "epoch": epoch+1,
                    "accuracy": test_accuracy,
                    "train_loss": train_loss,
                    "test_loss": test_loss
                    }, f)
            if early_stopping.early_stop:
                accelerator.print("Early stopping")
                break


    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        path = os.path.join(path, "checkpoint")
        del_files(path)  # delete checkpoint files
        accelerator.print('success delete checkpoints')