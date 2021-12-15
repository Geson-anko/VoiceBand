import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from parsers import get_training_parser
from models import VoiceBand
from datasets import RandomSample
from utils import load_config,get_now
import torch
import torch.utils.data as DataUtil
import os

if __name__ == "__main__":
    parser = get_training_parser()
    args = parser.parse_args()
    config_file = args.config_file
    dataset_path = args.dataset
    on_memory = args.on_memory
    logdir = args.logdir
    logname = args.logname
    epochs = args.epochs
    num_gpus= args.num_gpus
    num_nodes = args.num_nodes
    precision=args.precision
    save_name = args.save_name
    save_dir = args.save_dir
    num_workers = args.num_workers
    max_data_length = args.max_length
    view_interval = args.view_interval
    log_every_n_steps=args.log_every_n_steps

    config = load_config(config_file)
    batch_size = config.batch_size
    dataset = RandomSample(config,dataset_path,on_memory,True,max_length=max_data_length)
    data_loader = DataUtil.DataLoader(dataset, batch_size,shuffle=True,num_workers=num_workers)

    # set logger
    logger = pl_loggers.TensorBoardLogger(logdir,logname)
    trainer = pl.Trainer(logger=logger,gpus=num_gpus,num_nodes=num_nodes,precision=precision,max_epochs=epochs,log_every_n_steps=log_every_n_steps)
    
    # model
    model = VoiceBand(config)
    model.set_view_interval(view_interval)
    if save_name is None:
        save_name = model._get_name()
    trainer.fit(model,data_loader)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    name = save_name + get_now() + ".pth"
    path = os.path.join(save_dir,name)
    torch.save(model.state_dict(), path)
    print("Saved to ",path)
