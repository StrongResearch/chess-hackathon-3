import torch
import torch.nn as nn
from model import EasyLeela
torch.backends.cudnn.benchmark = True
from argparse import ArgumentParser
from pathlib import Path
from new_data_pipeline import multiprocess_generator
from threading import Thread
from queue import Queue


def queued_generator(queue, **kwargs):
    generator = multiprocess_generator(**kwargs)
    for batch in generator:
        batch = [torch.from_numpy(tensor) for tensor in batch]
        #batch = [torch.from_numpy(tensor).pin_memory() for tensor in batch]
        queue.put(batch)


class LeelaDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, **kwargs
    ):
        self.queue = Queue(maxsize=4)
        kwargs['queue'] = self.queue
        self.thread = Thread(target=queued_generator, kwargs=kwargs, daemon=True)
        self.thread.start()

    def __iter__(self):
        # worker_info = torch.utils.data.get_worker_info()
        # if worker_info is not None:
        #     raise RuntimeError("This dataset does multiprocessing internally, and should only have a single torch worker!")
        return self

    def __next__(self):
        return self.queue.get(block=True)

def train(model, dataset, args):
    breakpoint()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, pin_memory=False)

    optimizer = torch.optim.Adam(model.parameters())

    loss_function = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # train loop
    model.train()
    for epoch in range(args.max_epochs):
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(input) #TODO: check outputs 
            loss = loss_function(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{args.max_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

            # Check if we've reached the limit_train_batches
            if batch_idx >= args.limit_train_batches - 1:
                break
        
        torch.save(model.state_dict(), f"{args.save_dir}/model_epoch_{epoch+1}.pth")
        



class Driver:
    def __init__(self, data_folder='/data', model_save_folder='model_weights'):
        self.data_folder = data_folder
        self.model_save_folder = model_save_folder

    def main(self, args, should_train: bool):
        with torch.no_grad():
            model = EasyLeela(
                num_filters=args.num_filters,
                num_residual_blocks=args.num_residual_blocks,
                se_ratio=args.se_ratio,
                policy_loss_weight=args.policy_loss_weight,
                value_loss_weight=args.value_loss_weight,
                learning_rate=args.learning_rate
            )
            if should_train:
                dataset = LeelaDataset(
                    chunk_dir=args.dataset_path,
                    batch_size=args.batch_size,
                    skip_factor=args.skip_factor,
                    num_workers=args.num_workers,
                    shuffle_buffer_size=args.shuffle_buffer_size,
                )

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
    parser.add_argument('--max_epochs', type=int, default=100)
    args = parser.parse_args()
    driver.main(args, True)