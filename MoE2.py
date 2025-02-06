import os 
import torch
import torch.nn as nn
from tdc.benchmark_group import admet_group
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.data_loader import SMILESDataset 
from utils.utli import Scheduler, EarlyStopping
from utils.loss_fn import FocalLoss, HuberLoss
from model.moe import MoE, MoE_regression
import wandb

PROJECT_NAME = "ADMET_MoE_Topk_Project"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_evaluate(transformer_kinds,
                       benchmark_key,
                       task,
                       log_scale,
                       seeds=[1,2,3,4,5],
                       epochs=200,
                       batch_size=32,
                       learning_rate=1e-4):
    
     
    run_name = f"MoE_{benchmark_key}_{task}"
    
   
    wandb.init(project=PROJECT_NAME, name=run_name, config={
        "transformer_kinds": transformer_kinds,
        "benchmark": benchmark_key,
        "task": task,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seeds": seeds
    })
    


    group = admet_group(path="data/")
    benchmark = group.get(benchmark_key)
    benchmark_name = benchmark["name"]
    
    predictions_list = []
    os.makedirs(".ckpt", exist_ok=True)

    for seed in tqdm(seeds, desc=f"[{benchmark_key} | {task} | AdvancedMoE] Seeds"):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        train_df, test_df = benchmark["train_val"], benchmark["test"]
        train_df, valid_df = group.get_train_valid_split(benchmark=benchmark_name, split_type="default", seed=seed)
        
        
        if task == "regression" and log_scale:
            train_df["Y"] = train_df["Y"].apply(lambda x: np.log(x + 1))
            valid_df["Y"] = valid_df["Y"].apply(lambda x: np.log(x + 1))
            test_df["Y"] = test_df["Y"].apply(lambda x: np.log(x + 1))

        train_dataset = SMILESDataset(train_df)
        valid_dataset = SMILESDataset(valid_df)
        test_dataset = SMILESDataset(test_df)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if task == "binary":
            model = MoE(transformer_kinds=transformer_kinds)
            criterion = FocalLoss(alpha=0.25, gamma=2, reduction='mean')
        else:
            model = MoE_regression(transformer_kinds= transformer_kinds)
            criterion = HuberLoss(reduction='mean')

        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = Scheduler(optimizer, factor=0.5, patience=5)
        ckpt_path = f".ckpt/advanced_checkpoint_{benchmark_name}_{seed}.pt"
        early_stopper = EarlyStopping(patience=10, delta=0.001, path=ckpt_path)
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_total = 0
            train_correct = 0
            
            for smiles_batch, labels_batch in train_loader:
                smiles_batch = list(smiles_batch)
                labels_batch = labels_batch.float().unsqueeze(1).to(device)
                
                optimizer.zero_grad()
                y_pred, _ = model(smiles_batch)
                loss = criterion(y_pred, labels_batch) 
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                bs = labels_batch.size(0)
                train_loss += loss.item() * bs
                train_total += bs
                if task == "binary":
                    preds = (y_pred > 0.5).float()
                    train_correct += (preds == labels_batch).sum().item()
            
            train_loss_epoch = train_loss / train_total
            if task == "binary":
                train_acc_epoch = train_correct / train_total
                wandb.log({"train_loss": train_loss_epoch, "train_acc": train_acc_epoch, "epoch": epoch})
            else:
                train_acc_epoch = None


            model.eval()
            val_loss = 0.0
            val_total = 0
            val_correct = 0
            all_labels = []
            all_preds = []
            
            with torch.no_grad():
                for smiles_batch, labels_batch in valid_loader:
                    smiles_batch = list(smiles_batch)
                    labels_batch = labels_batch.float().unsqueeze(1).to(device)
                    y_pred, moe_loss = model(smiles_batch)
                    loss = criterion(y_pred, labels_batch) 
                    val_loss += loss.item() * labels_batch.size(0)
                    val_total += labels_batch.size(0)
                    
                    label_cpu = labels_batch.cpu().numpy().flatten()
                    pred_cpu = y_pred.cpu().numpy().flatten()
                    all_labels.extend(label_cpu)
                    all_preds.extend(pred_cpu)
                    
                    if task == "binary":
                        preds = (y_pred > 0.5).float()
                        val_correct += (preds == labels_batch).sum().item()
            
            val_loss_epoch = val_loss / val_total
            if task == "binary":
                val_acc_epoch = val_correct / val_total
                try:
                    from sklearn.metrics import average_precision_score
                    val_auprc = average_precision_score(all_labels, all_preds)
                except ImportError:
                    val_auprc = 0.0
                val_metric = val_auprc
            else:
                if log_scale:
                    exp_labels = np.exp(np.array(all_labels)) - 1
                    exp_preds = np.exp(np.array(all_preds)) - 1
                    from sklearn.metrics import mean_absolute_error
                    val_mae = mean_absolute_error(exp_labels, exp_preds)
                else:
                    from sklearn.metrics import mean_absolute_error
                    val_mae = mean_absolute_error(all_labels, all_preds)
                val_metric = val_mae
            

            if task == "binary":
                print(f"[Epoch {epoch+1}/{epochs} | Seed {seed}] "
                      f"Train Loss: {train_loss_epoch:.4f}, Val Loss: {val_loss_epoch:.4f}, "
                      f"Train Acc: {train_acc_epoch:.4f}, Val Acc: {val_acc_epoch:.4f} "
                      f"Val AUPRC: {val_metric:.4f}")
                wandb.log({"val_loss": val_loss_epoch, "val_auprc": val_metric,"val_acc":val_acc_epoch, "epoch": epoch})
            else:
                print(f"[Epoch {epoch+1}/{epochs} | Seed {seed}] "
                      f"Train Loss: {train_loss_epoch:.4f}, Val Loss: {val_loss_epoch:.4f}, "
                      f"Val MAE: {val_metric:.4f}")
                wandb.log({"train_loss":train_loss_epoch, "val loss":val_loss_epoch,"val mae": val_metric, "epoch": epoch})
        
            scheduler.step(val_loss_epoch)
            early_stopper(val_loss_epoch, model)
            if early_stopper.early_stop:
                print(f"Early stopping at epoch {epoch+1} for seed {seed}")
                break

        model.load_state_dict(torch.load(ckpt_path))
        model.eval()
        test_preds = []
        with torch.no_grad():
            for smiles_batch, _ in test_loader:
                smiles_batch = list(smiles_batch)
                y_pred, __builtins__ = model(smiles_batch)
                test_preds.extend(y_pred.cpu().numpy().flatten())
        test_preds = np.array(test_preds)
        predictions = {}
        if task == "binary":
            predictions[benchmark_name] = torch.tensor(test_preds)
        else:
            if log_scale:
                predictions[benchmark_name] = torch.tensor(np.exp(test_preds) - 1)
            else:
                predictions[benchmark_name] = torch.tensor(test_preds)
        predictions_list.append(predictions)
 
        predictions_list.append(predictions)
 
    results = group.evaluate_many(predictions_list)
    final_score = results.get(benchmark_name, None)
    
    if final_score:
        score_mean, score_std = final_score
        if task == "binary":
            wandb.log({"test_auprc_mean": score_mean, "test_auprc_std": score_std})
            print(f"[TEST] {benchmark_key} => AUPRC mean={score_mean:.4f}, std={score_std:.4f}")
        else:
            wandb.log({"test_mae_mean": score_mean, "test_mae_std": score_std})
            print(f"[TEST] {benchmark_key} => MAE mean={score_mean:.4f}, std={score_std:.4f}")

    wandb.finish()



if __name__ == "__main__":
    transformer_kinds = [
        "gin_supervised_infomax",
        "gin_supervised_contextpred",
        "gin_supervised_edgepred",
        "gin_supervised_masking"
    ]

    benchmark_config = {
       # 'caco2_wang': ('regression', False),
         #'bioavailability_ma': ('binary', False),
       # 'lipophilicity_astrazeneca': ('regression', False),
       # 'solubility_aqsoldb': ('regression', False),
         'hia_hou': ('binary', False),
         'pgp_broccatelli': ('binary', False),
         'bbb_martins': ('binary', False),
       # 'ppbr_az': ('regression', False),
       # 'vdss_lombardo': ('regression', True),
       # 'cyp2c9_veith': ('binary', False),
       # 'cyp2d6_veith': ('binary', False),
       # 'cyp3a4_veith': ('binary', False),
        'cyp2c9_substrate_carbonmangels': ('binary', False),
        'cyp2d6_substrate_carbonmangels': ('binary', False),
        'cyp3a4_substrate_carbonmangels': ('binary', False),
       # 'half_life_obach': ('regression', True),
       # 'clearance_hepatocyte_az': ('regression', True),
       # 'clearance_microsome_az': ('regression', True),
       # 'ld50_zhu': ('regression', False),
       # 'herg': ('binary', False),
       # 'ames': ('binary', False),
       # 'dili': ('binary', False)
    }
    
    seeds = [1, 2, 3, 4, 5]
    epochs = 200
    batch_size = 32
    learning_rate = 1e-3
    
    for benchmark_key, (task, log_scale) in benchmark_config.items():
        train_and_evaluate(transformer_kinds=transformer_kinds,
                           benchmark_key=benchmark_key,
                           task=task,
                           log_scale=log_scale,
                           seeds=seeds,
                           epochs=epochs,
                           batch_size=batch_size,
                           learning_rate=learning_rate)

    print("All runs done!")
