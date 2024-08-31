import torch
import torch.nn as nn
from torch.nn import functional as F
from model import EasyLeela
torch.backends.cudnn.benchmark = True
from argparse import ArgumentParser
from pathlib import Path
from new_data_pipeline import multiprocess_generator
from threading import Thread
from queue import Queue
import sys 
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from hackathon_train.dataset.leela_dataset import LeelaDataset


def queued_generator(queue, **kwargs):
    generator = multiprocess_generator(**kwargs)
    for batch in generator:
        batch = [torch.from_numpy(tensor) for tensor in batch]
        #batch = [torch.from_numpy(tensor).pin_memory() for tensor in batch]
        queue.put(batch)

def policy_loss(target: torch.Tensor, output: torch.Tensor):
    # Illegal moves are marked by a value of -1 in the labels - we mask these with large negative values
    output.masked_fill_(target < 0, -1e4)
    # The large negative values will still break the loss, so we replace them with 0 once we finish masking
    target = F.relu(target)
    log_prob = F.log_softmax(output, dim=1)
    nll = -(target * log_prob).sum() / output.shape[0]
    return nll


def value_loss(target: torch.Tensor, output: torch.Tensor):
    log_prob = F.log_softmax(output, dim=1)
    value_nll = -(target * log_prob)
    return value_nll.sum() / output.shape[0]



def train(model, dataset, args):
    print("Entered train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, pin_memory=False)

    optimizer = torch.optim.Adam(model.parameters())

    loss_function = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # train loop
    model.train()
    for epoch in range(args.max_epochs):
        epoch_loss = 0.
        for batch_idx, output in enumerate(dataloader):
            inputs, policy, z, orig_q, ply_count = output[0],output[1],output[2],output[3],output[4] 
            inputs, policy, z, orig_q, ply_count = inputs.to(device), policy.to(device), z.to(device), orig_q.to(device), ply_count.to(device)
            #change the inputs
            (policy_out, value_out) = model(inputs)
            
            loss = 0.5*nn.CrossEntropyLoss()(orig_q, value_out) + 0.5*nn.MSELoss()(policy, policy_out)  
            #loss = 0.5*policy_loss(policy, policy_out) + 0.5*value_loss(orig_q, value_out)
            epoch_loss +=loss.item()
            optimizer.zero_grad()
            loss.backward()
            if torch.isnan(loss):
                print(f"NaN loss detected. policy_out: {policy_out}, value_out: {value_out}")
                print(f"policy: {policy}, orig_q: {orig_q}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Check if we've reached the limit_train_batches
            #if batch_idx >= args.limit_train_batches - 1:
            #    break
        
        avg_epoch_loss = epoch_loss / (batch_idx + 1)
        print(f"Epoch {epoch+1}/{args.max_epochs} completed. Average Loss: {avg_epoch_loss:.4f}")

        
        torch.save(model.state_dict(), f"{args.save_dir}/model_epoch_{epoch+1}.pth")
        



class Driver:
    def __init__(self, data_folder='/data', model_save_folder='model_weights'):
        self.data_folder = data_folder
        self.model_save_folder = model_save_folder

    def main(self, args, should_train: bool):
        model = EasyLeela(
                num_filters=args.num_filters,
                num_residual_blocks=args.num_residual_blocks,
                se_ratio=args.se_ratio,
                policy_loss_weight=args.policy_loss_weight,
                value_loss_weight=args.value_loss_weight,
                learning_rate=args.learning_rate
        )
        if should_train:
            print("Before dataset")
            dataset = LeelaDataset()

            train(model, dataset, args)
            print("Finished training.")
            
        else:
            raise NotImplementedError("Have yet to upload")


    

if __name__ == "__main__":
    driver = Driver()
    parser = ArgumentParser()
    # These parameters control the net and the training process
    parser.add_argument("--num_filters", type=int, default=128)
    parser.add_argument("--num_residual_blocks", type=int, default=2)
    parser.add_argument("--se_ratio", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_grad_norm", type=float, default=5.6)
    parser.add_argument("--mixed_precision", action="store_true")
    # These parameters control the data pipeline
    parser.add_argument("--dataset_path", type=Path, default='/data/training-run1-test80-20220711-0217.tar')
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--shuffle_buffer_size", type=int, default=2 ** 19)
    parser.add_argument("--skip_factor", type=int, default=32)
    parser.add_argument("--save_dir", type=Path, default='model_weights/')
    # These parameters control the loss calculation. They should not be changed unless you
    # know what you're doing, as the loss values you get will not be comparable with other
    # people's unless they are kept at the defaults.
    parser.add_argument("--policy_loss_weight", type=float, default=1.0)
    parser.add_argument("--value_loss_weight", type=float, default=1.6)
    #These parameters control the training
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--limit_train_batches', type=int, default=1000)
    args = parser.parse_args()
    driver.main(args, True)