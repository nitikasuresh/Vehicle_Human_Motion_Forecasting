# -*- coding: utf-8 -*-
import argparse
import os
import time
from turtle import forward
from sklearn.cluster import KMeans
import wandb
USE_WANDB = True
if USE_WANDB:
    wandb.init(project="motion_prediction", reinit=True)
# wandb.init(project="motion-prediction")
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from config.Config import Config
# ============= Dataset =====================
from lib.dataset.collate import collate_single_cpu
from lib.dataset.dataset_for_argoverse import STFDataset as ArgoverseDataset
# ============= Models ======================
from lib.models.mmTransformer import mmTrans
from lib.utils.evaluation_utils import FormatDataTensor, compute_forecasting_metrics, FormatData


from lib.utils.utilities import load_checkpoint, load_model_class, save_checkpoint

from lib.models.mmTransformer import MLP
def parse_args():

    parser = argparse.ArgumentParser(description='Train the mmTransformer')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--model-name', type=str, default='demo_train')
    parser.add_argument('--model-save-path', type=str, default='./models/')
    parser.add_argument('--loss', type=str, default='nll')

    args = parser.parse_args()
    return args

class SimpleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self,results):
        forecasted_trajectories = results['forecasted_trajectories']
        gt_trajectories = results['gt_trajectories']
        forecasted_probabilities = results['forecasted_probabilities']
        pred = []
        gt = []
        confidences = []
        for label, gtu in gt_trajectories.items():
            max_num_traj = min(6, len(forecasted_trajectories[label]))
            if forecasted_probabilities is not None:
                sorted_idx = torch.argsort(torch.Tensor(forecasted_probabilities[label]),descending=True)
                pruned_probabilities = [forecasted_probabilities[label][t] for t in sorted_idx[:max_num_traj]]

                # Normalize the probabilites
                prob_sum = np.sum(pruned_probabilities)
                prunded_conf = [p / prob_sum for p in pruned_probabilities]
            else:
                sorted_idx = np.arange(len(forecasted_trajectories[label]))
            prunedTraj = [forecasted_trajectories[label][t] for t in sorted_idx[:max_num_traj]]
            gt.append(gtu[None,:])
            pred.append(torch.stack(prunedTraj,dim=0))
            confidences.append(torch.stack(prunded_conf,dim=0))
        # https://github.com/JerryIshihara/lyft-motion-prediction-for-autonomous-vehicle/blob/2a533046f04392f213eeb969e2ec4162a0ce3852/train.py#L149
        pred = torch.stack(pred,dim=0)
        confidences = torch.stack(confidences,dim=0)
        gt = torch.stack(gt,dim=0)

        #NLL loss
        error = torch.sum(((gt - pred)) ** 2, dim=-1)
        with np.errstate(divide="ignore"):
            error = torch.log(confidences) - 0.5 * \
                    torch.sum(error, dim=-1) 
        max_value, _ = error.max(dim=1, keepdim=True)
        error = -torch.log(torch.sum(torch.exp(error - max_value),
                                 dim=-1, keepdim=True)) - max_value
        return torch.mean(error)


class MultiLoss(nn.Module):
    def __init__(self):
        super(MultiLoss, self).__init__()
  
        # weights for loss
        self.sigma = nn.Parameter(torch.ones(3))
        self.mse = nn.MSELoss(reduction = 'sum')
        self.regLoss = nn.SmoothL1Loss(reduction = 'sum')
        self.confLoss = nn.KLDivLoss(reduction="none", log_target=True)
     

    def forward(self, results):
        forecasted_trajectories = results['forecasted_trajectories']
        gt_trajectories = results['gt_trajectories']
        forecasted_probabilities = results['forecasted_probabilities']
        regressionLossList = []
        confidenceLossList = []
        
        # sort through gt trajectories
        for k, v in gt_trajectories.items():
            # max number of guesses is 6
            max_num_traj = min(6, len(forecasted_trajectories[k]))
            if forecasted_probabilities is not None:
                sorted_idx = torch.argsort(torch.Tensor(forecasted_probabilities[k]),descending=True)
                pruned_probabilities = torch.stack([forecasted_probabilities[k][t] for t in sorted_idx[:max_num_traj]]) #:max_num_traj
                # Normalize the probabilites
                prob_sum = torch.sum(pruned_probabilities)
                pruned_probabilities = torch.stack([p / prob_sum for p in pruned_probabilities])
            else:
                sorted_idx = torch.arange(len(forecasted_trajectories[k]))
            pruned_trajectories = torch.stack([forecasted_trajectories[k][t] for t in sorted_idx[:max_num_traj]]) #:max_num_traj

            # get regression loss and get confidence loss
            tau = []
            dists = []
            for p in pruned_trajectories:
                losreg = self.regLoss(p, v)
                regressionLossList.append(losreg)
                tau.append(self.distance_metric(p, v).flatten())
                dists.append(-self.mse(p, v).flatten())
                
            tau = torch.stack(tau)
            tau = F.log_softmax((tau), dim=0)
            
            dists = torch.stack(dists)
            lambda_s = F.log_softmax(-(dists), dim=0)
            confidenceLoss = self.confLoss(lambda_s,tau)
            confidenceLossList.append(confidenceLoss)
 

        regressionLossList = torch.stack(regressionLossList,dim=0)
        confidenceLossList = torch.stack(confidenceLossList,dim=0) 

        loss = torch.mean(regressionLossList) + 0.1*torch.mean(confidenceLossList)
        return loss

    # from https://github.com/Henry1iu/TNT-Trajectory-Predition/blob/e73b10847c56ab1f632f62e223eb51467d95625b/core/model/layers/scoring_and_selection.py#L10
    def distance_metric(self, traj_candidate, traj_gt): #: torch.Tensor
        """
            compute the distance between the candidate trajectories and gt trajectory
        :param traj_candidate: torch.Tensor, [batch_size, M, horizon * 2] or [M, horizon * 2]
        :param traj_gt: torch.Tensor, [batch_size, horizon * 2] or [1, horizon * 2]
        :   return: distance, torch.Tensor, [batch_size, M] or [1, M]
        """
 
        if traj_candidate.dim() == 2:
            traj_candidate = traj_candidate.unsqueeze(1) 

        _, M, _ = traj_candidate.size()
        horizon_2_times = 30*2
        dis = torch.pow(traj_candidate - traj_gt.unsqueeze(1), 2).view(-1, M, int(horizon_2_times / 2), 2)

        dis, _ = torch.max(torch.sum(dis, dim=3), dim=2)

        return dis

if __name__ == "__main__":
    with torch.cuda.device(2):
        start_time = time.time()
        gpu_num = torch.cuda.device_count()
        print("gpu number:{}".format(gpu_num))

        args = parse_args()
        cfg = Config.fromfile(args.config)

        # ================================== INIT DATASET ==========================================================
        train_cfg = cfg.get('train_dataset')
        train_dataset = ArgoverseDataset(train_cfg)
        train_dataloader = DataLoader(train_dataset,
                                    shuffle=train_cfg["shuffle"],
                                    batch_size=train_cfg["batch_size"],
                                    num_workers=train_cfg["workers_per_gpu"],
                                    collate_fn=collate_single_cpu)
        validation_cfg = cfg.get('val_dataset')
        val_dataset = ArgoverseDataset(validation_cfg)
        val_dataloader = DataLoader(val_dataset,
                                    shuffle=validation_cfg["shuffle"],
                                    batch_size=validation_cfg["batch_size"],
                                    num_workers=validation_cfg["workers_per_gpu"],
                                    collate_fn=collate_single_cpu)
        # =================================== Metric Initial =======================================================
        format_results_val = FormatData()
        evaluate = partial(compute_forecasting_metrics,
                        max_n_guesses=6,
                        horizon=30,
                        miss_threshold=2.0)
        # =================================== INIT MODEL ===========================================================
        model_cfg = cfg.get('model')
        stacked_transfomre = load_model_class(model_cfg['type'])
        model = mmTrans(stacked_transfomre, model_cfg).cuda()
    
        print('Finished Initialization in {:.3f}s!!!'.format(
            time.time()-start_time))
        ############################### Define Loss
        if args.loss == "multiloss":
            criterion = MultiLoss().cuda()
        elif args.loss == "nll":
            criterion = SimpleLoss().cuda()

        optimizer = optim.AdamW([{'params': model.parameters()},{'params': criterion.parameters()}], lr=0.001, weight_decay=0.0001)#maximize=True
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        # ==================================== Training LOOP =====================================================
        model.train()
        progress_train_bar = tqdm(train_dataloader)
        progress_bar = tqdm(val_dataloader)
        iteration = 10
        iteration_for = 500
        num_epochs = 100
        minAde = np.Inf
        for epoch in range(num_epochs):
            for j, data in enumerate(progress_train_bar):
                for key in data.keys():
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].cuda()
                optimizer.zero_grad()
                out = model(data)
                # format results and compute loss
                format_results = FormatDataTensor()
                results = format_results(data, out)
                lossitem = criterion(format_results.results)
                lossitem.backward()
                # Optimizer takes one step
                optimizer.step()
                if j % iteration_for == 0:
                    if USE_WANDB:
                        wandb.log({"epoch":epoch,"Loss ":lossitem.item()})
                    print("Training Results")
                    print(lossitem.item())
            scheduler.step()
            if epoch % iteration == 0:
                model_name = os.path.join(args.model_save_path,
                                '{}_at_epoch_{}.pt'.format(args.model_name, str(epoch)))
                save_checkpoint(model_name,model,optimizer,lossitem.item())
                # ==================================== EVALUATION LOOP =====================================================
                model.eval()
                with torch.no_grad():
                    for j, dataj in enumerate(progress_bar):
                        for key in dataj.keys():
                            if isinstance(dataj[key], torch.Tensor):
                                dataj[key] = dataj[key].cuda()
                        out = model(dataj)
                        format_results_val(dataj, out)
                res = evaluate(**format_results_val.results)
                if USE_WANDB:
                    wandb.log({"epoch":epoch,"minAde ":res["minADE"], "minFDE": res["minFDE"], "MR": res["MR"]})
                if res["minADE"] < minAde:
                    # save the best results
                    model_name = os.path.join(args.model_save_path,
                                '{}_best.pt'.format(args.model_name))
                    save_checkpoint(model_name,model,optimizer,res["minADE"])
                    minAde = res["minADE"]
                format_results_val = FormatData()
                print('Validation Process Finished!!')
                model.train()

        model_name = os.path.join(args.model_save_path,
                                '{}_at_epoch_{}.pt'.format(args.model_name, str(epoch)))
        save_checkpoint(model_name,model,optimizer,lossitem.item())
        print("Training Results")
        print(lossitem.item())
        # ==================================== EVALUATION LOOP =====================================================
        model.eval()
        with torch.no_grad():
            for j, dataj in enumerate(progress_bar):
                for key in dataj.keys():
                    if isinstance(dataj[key], torch.Tensor):
                        dataj[key] = dataj[key].cuda()
                out = model(dataj)
                format_results_val(dataj, out)
        res = evaluate(**format_results_val.results)
        if USE_WANDB:
            wandb.log({"epoch":epoch+1,"minAde ":res["minADE"], "minFDE": res["minFDE"], "MR": res["MR"]})
        if res["minADE"] < minAde:
            # save the best results
            model_name = os.path.join(args.model_save_path,
                            '{}_best.pt'.format(args.model_name))
            save_checkpoint(model_name,model,optimizer,res["minADE"])
            minAde = res["minADE"]
        format_results_val = FormatData()
        print('Run Finished!!')


        
