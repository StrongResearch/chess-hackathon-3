import torch
import torch.nn as nn
torch.backends.cudnn.benchmark = True


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

class Driver:
    def __init__(self, data_folder='/data', model_save_folder='model_weights'):
        self.data_folder = data_folder
        self.model_save_folder = model_save_folder

    def main(self, should_train: bool):
        with torch.no_grad():
            model = 
            if should_train:


            if args.save_dir.is_dir():
                try:
                    model = LeelaZeroNet.load_from_checkpoint(args.save_dir)
                except:
                    model = None
            if model is None:
                model = LeelaZeroNet(
                    num_filters=args.num_filters,
                    num_residual_blocks=args.num_residual_blocks,
                    se_ratio=args.se_ratio,
                    policy_loss_weight=args.policy_loss_weight,
                    value_loss_weight=args.value_loss_weight,
                    moves_left_loss_weight=args.moves_left_loss_weight,
                    q_ratio=args.q_ratio,
                    optimizer=args.optimizer,
                    learning_rate=args.learning_rate
                )

            dataset = LeelaDataset(
                chunk_dir=args.dataset_path,
                batch_size=args.batch_size,
                skip_factor=args.skip_factor,
                num_workers=args.num_workers,
                shuffle_buffer_size=args.shuffle_buffer_size,
            )

            dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, pin_memory=True)

            trainer = pl.Trainer(accelerator="gpu", gpus=1, precision=precision, limit_train_batches=8192, max_epochs=1,
                                default_root_dir=args.save_dir)
            trainer.fit(model, dataloader)


    

if __name__ == "__main__":
    driver = Driver()
    driver.main()
