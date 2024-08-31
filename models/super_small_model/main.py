import torch
import torch.nn as nn
from torch.nn import functional as F
from model import SimpleCNN, BiggerNet, init_weights
<<<<<<< HEAD
from model import SimpleCNN, ZeroNet
=======
>>>>>>> 9025686 (add init weight)
torch.backends.cudnn.benchmark = True
from argparse import ArgumentParser
from pathlib import Path
from threading import Thread
from queue import Queue
import sys 
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from models.small_model.new_data_pipeline import multiprocess_generator
from hackathon_train.dataset.leela_dataset import  LeelaDataset
import logging 
from pathlib import Path
from utils import split_files
from tqdm import tqdm 




def policy_loss(pred, target, valid_move_weight=0.75):
    '''
    target: tensor of size (bs, 1858) and contains mostly -1s. Other values are in 1
    '''
    non_negative_idxs = torch.where(target >= 0)
    negative_idxs = torch.where(target < 0)
    non_negative_preds = pred[non_negative_idxs]
    non_negative_targets = target[non_negative_idxs]
    negative_preds = pred[negative_idxs]
    negative_targets = target[negative_idxs]

    mse = nn.MSELoss()
    loss = valid_move_weight*mse(non_negative_preds, non_negative_targets) + (1-valid_move_weight)*mse(negative_preds, negative_targets)
    return loss 

class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value_pred, y_policy, policy_pred):
        mse_crit = torch.nn.MSELoss()
        policy_loss = mse_crit(y_policy, policy_pred)
        
        value_loss = mse_crit(y_value, value_pred) 
        
        total_error = policy_loss + value_loss
        return total_error


def setup_logging(log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO)

def train(model, train_dataset, val_dataset, args):
    setup_logging(f"{args.save_dir}/training_log.txt")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_bs, pin_memory=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.train_bs, pin_memory=False)
    
    optimizer = torch.optim.Adam(model.parameters())

    loss_function = AlphaLoss()
    mse_loss = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_loss = float('inf')

    # train loop

    for epoch in range(args.max_epochs):
        model.train()
        train_loss = 0.0
        train_steps = 0 
        for batch_idx, output in enumerate(train_dataloader):
            inputs, policy, z, orig_q, ply_count = output[0],output[1],output[2],output[3],output[4] 
            inputs, policy, z, orig_q, ply_count = inputs.to(device), policy.to(device), z.to(device), orig_q.to(device), ply_count.to(device)
          
            #TODO: if small cnn model, change the inputs
            policy_pred, value_pred = model(inputs)
            #loss = loss_function(orig_q, value_pred, policy, policy_pred)
            loss = mse_loss(policy, policy_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1
        
        avg_train_loss = train_loss / train_steps

        if epoch % args.val_freq != 0:
            log_message = f"Epoch: {epoch + 1} | Train Loss: {avg_train_loss:.6f}"
            print(log_message)
            logging.info(log_message)
        
        ### Val loss step ###
        if epoch > 0 and epoch % args.val_freq == 0:
            model.eval()
            val_loss = 0.0
            val_steps = 0 

            with torch.no_grad():
                for batch_idx, output in enumerate(val_dataloader):
                    inputs, policy, z, orig_q, ply_count = output[0],output[1],output[2],output[3],output[4] 
                    inputs, policy, z, orig_q, ply_count = inputs.to(device), policy.to(device), z.to(device), orig_q.to(device), ply_count.to(device)

                    policy_pred, value_pred = model(inputs)
                    #loss = loss_function(orig_q, value_pred, policy, policy_pred)      
                    loss = mse_loss(policy, policy_pred)

                    val_loss += loss.item()
                    val_steps += 1

            avg_val_loss = val_loss / val_steps 
            log_message = f"Epoch: {epoch + 1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
            print(log_message)
            logging.info(log_message)

            # save best model 
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f"{args.save_dir}/best_model_loss_{best_val_loss}.pth")
                print("Saved best model.")

    print("Training completed.")        
        

def test(model, test_dataset,  args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.train_bs)
    total_loss = 0.0
    
    loss_function = AlphaLoss()
    mse_loss = nn.MSELoss()
    #loss_function = torch.nn.MSELoss()
    with torch.no_grad():
        batch_step = 0
        for batch_idx, output in tqdm(enumerate(test_dataloader)):
            inputs, policy, z, orig_q, ply_count = [item.to(device) for item in output]

            policy_pred, value_pred = model(inputs)
            #loss = loss_function(orig_q, value_pred, policy, policy_pred)
            loss = mse_loss(policy, policy_pred)

            total_loss += loss.item()
            batch_step += 1
    
    avg_test_loss = total_loss / batch_step
    print(f"Avg test loss is {avg_test_loss:.6f}")
    return avg_test_loss






class Driver:
    def __init__(self, data_folder='/data', model_save_folder='model_weights'):
        self.data_folder = data_folder
        self.model_save_folder = model_save_folder

    def main(self, args, should_train: bool):
        model_weight_path = 'models/super_small_model/model_weights/model_epoch_60.pth'

        #model = ZeroNet(num_res_blocks=0)
        #model = SimpleCNN() #CAN SWAP MODEL HERE
        model = BiggerNet()
        model.apply(init_weights)
        files = list(Path(args.dataset_path).glob("*"))
        train_files, val_files, test_files = split_files(files)
       
        train_dataset = LeelaDataset(tar_files=train_files)
        val_dataset = LeelaDataset(tar_files=val_files)
        test_dataset = LeelaDataset(tar_files=test_files)

        if should_train:
            train(model, train_dataset, val_dataset, args)
            print("Finished training.")
            
        else: #load model 
            model.load_state_dict(torch.load(model_weight_path))
   
        eval_loss = test(model, test_dataset, args)
    

if __name__ == "__main__":
    driver = Driver()
    parser = ArgumentParser()
    # These parameters control the data pipeline
    parser.add_argument("--dataset_path", type=Path, default='/home/justin/Desktop/Code/chess-hackathon-3/small_data')
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--shuffle_buffer_size", type=int, default=2 ** 19)
    parser.add_argument("--skip_factor", type=int, default=32)
    parser.add_argument("--save_dir", type=Path, default='models/super_small_model/model_weights')
    # These parameters control the loss calculation. They should not be changed unless you
    # know what you're doing, as the loss values you get will not be comparable with other
    # people's unless they are kept at the defaults.
    parser.add_argument("--policy_loss_weight", type=float, default=1.0)
    parser.add_argument("--value_loss_weight", type=float, default=1.6)
    #These parameters control the training
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--train_bs', type=int, default=32)
    parser.add_argument('--val_freq', type=int, default=5)
    args = parser.parse_args()
    driver.main(args, True)