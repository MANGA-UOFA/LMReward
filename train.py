import torch
import transformers
import argparse
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import itertools
from contextlib import nullcontext
from utils.logger import Logger, AvgStat
import random
import numpy as np
from utils.dataset import DataCollector
from utils.dataset import DatasetForSeq2Seq, BatchedDataset


def shift_right(tensor, decoder_start_token_id):
    assert len(tensor.shape) == 2
    start_ids = torch.ones((tensor.shape[0], 1)).to(
        tensor) * decoder_start_token_id
    return torch.cat([start_ids, tensor[..., :-1]], dim=-1)


logger = Logger()


class TemperatureCrossEntropy(torch.nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        if 'temperature' in kwargs:
            self.temperature = kwargs.pop('temperature')
        else:
            self.temperature = 1.0
        super().__init__(*args, **kwargs)

    def forward(self, input, target):
        return super().forward(input * self.temperature, target)


class FP32Scaler(torch.cuda.amp.GradScaler):
    """
    FP32Scaler is for compatability with AMPScaler.
    But it also automatically checks gradient overflow.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        def get_grad_norm(parameters, norm_type=2.0):
            parameters = list(parameters)
            device = parameters[0].grad.device
            return torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters if p.grad is not None]), norm_type)
    
        parameters = itertools.chain(*[group['params'] for group in optimizer.param_groups])
        grad_norm = get_grad_norm(parameters)
        if grad_norm.isnan() or grad_norm.isinf():
            return
        return optimizer.step()

    def update(self):
        return

    def get_scale(self):
        return 1.0

    def unscale_(self, optimizer):
        return

class Trainer:
    def __init__(
        self,
        args,
        rank,
        dataset,
        model,
        optimizer_name,
        scheduler_name,
        criterion_name,
        optimizer_params,
        scheduler_params,
        criterion_params,
        update_per_save,
        update_per_log,
        iter_per_update,
        num_training_steps,
        save_dir,
        fp32,
    ):
        assert optimizer_name in ['adam']
        assert scheduler_name in ['linear']
        assert criterion_name in ['cross_entropy', 'temperature_cross_entropy']

        if optimizer_name == 'adam':
            optimizer_cls = torch.optim.Adam

        if scheduler_name == 'linear':
            scheduler_cls = transformers.get_linear_schedule_with_warmup

        if criterion_name == 'cross_entropy':
            criterion_cls = torch.nn.CrossEntropyLoss
        elif criterion_name == 'temperature_cross_entropy':
            criterion_cls = TemperatureCrossEntropy
        self.args = args
        self.rank = rank
        self.config = model.config
        self.model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_rank()],
            output_device=dist.get_rank()
        )
        self.dataset = dataset
        self.optimizer = optimizer_cls(
            self.model.parameters(), **optimizer_params)
        self.scheduler = scheduler_cls(self.optimizer, **scheduler_params)
        self.criterion = criterion_cls(**criterion_params)

        self.update_per_save = update_per_save
        self.update_per_log = update_per_log
        self.iter_per_update = iter_per_update
        self.num_training_steps = num_training_steps
        self.save_dir = save_dir
        self.fp32 = fp32

        self.iter_steps = 0
        self.update_steps = 0
        self.epoch = 1
        self.in_epoch_step = 0
        if self.fp32:
            self.scaler = FP32Scaler()
        else:
            self.scaler = torch.cuda.amp.GradScaler()

        epoch_lengths_per_gpu = [None] * dist.get_world_size()
        dist.all_gather_object(epoch_lengths_per_gpu, len(self.dataset))
        self.epoch_length = min(epoch_lengths_per_gpu)
        if dist.get_rank() == 0:
            logger.log(f'\t original lengths: {epoch_lengths_per_gpu}')
            logger.log(f'\t reduced to {self.epoch_length}')

        if os.path.exists(os.path.join(self.save_dir, 'model-last/status.pt')):
            if dist.get_rank() == 0:
                logger.log(
                    f'loading from {os.path.join(self.save_dir, "model-last")}')
            self.load_status(os.path.join(self.save_dir, 'model-last'))

    def train(self):
        device = torch.device(self.rank)
        loss_avg = AvgStat()
        grad_norm_avg = AvgStat()
        while self.update_steps < self.num_training_steps:
            dataiter = iter(self.dataset)
            for in_epoch_step in range(self.epoch_length):
                self.in_epoch_step = in_epoch_step + 1
                if self.update_steps >= self.num_training_steps:
                    break
                data = next(dataiter)
                if self.iter_steps % self.epoch_length != in_epoch_step:  # reloaded:
                    logger.log('skip')
                    continue
                self.model.train()
                enc_input = data['src'].to(device)
                dec_output = data['tgt'].to(device)
                dec_input_ids = shift_right(
                    dec_output['input_ids'], self.config.decoder_start_token_id)
                model_inputs = {
                    'input_ids': enc_input['input_ids'],
                    'attention_mask': enc_input['attention_mask'],
                    'decoder_input_ids': dec_input_ids,
                    'decoder_attention_mask': dec_output['attention_mask'],
                    'return_dict': True,
                    'use_cache': False,
                }
                ddp_context = nullcontext() if (self.iter_steps + 1) % self.iter_per_update == 0 else self.model.no_sync()
                amp_context = nullcontext() if self.fp32 else torch.cuda.amp.autocast()
                with ddp_context:
                    with amp_context:
                        output = self.model(**model_inputs)
                        logits = output.logits
                        loss = self.criterion(
                            logits.reshape(-1, logits.shape[-1]),
                            dec_output['input_ids'].reshape(-1),
                        )
                        loss = loss.masked_fill(dec_output['attention_mask'].reshape(-1).logical_not(), 0.0)
                        loss = loss.sum() / dec_output['attention_mask'].sum() # average by #tokens
                        loss_avg.take(loss.item())
                        loss = loss / self.iter_per_update # gradient accumulation
                        loss = loss * (dec_output['attention_mask'].sum() / self.args.max_tokens) # proportional to batch size
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.log({'warning': 'forward nan/inf detected'})
                    self.scaler.scale(loss).backward()
                self.iter_steps = self.iter_steps + 1
                
                if self.iter_steps % self.iter_per_update == 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm_avg.take(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm).item())
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.update_steps += 1
                    if dist.get_rank() == 0:
                        if self.update_steps % self.update_per_log == 0:
                            self.log({
                                'lr': self.scheduler.get_last_lr()[0],
                                'loss': loss_avg.pop(),
                                'scaling': self.scaler.get_scale(),
                                'grad_norm': grad_norm_avg.pop(),
                            })
                        if self.update_steps % self.update_per_save == 0:
                            self.save()
                del data
            self.epoch += 1
        del dataiter
        self.dataset.cleanup()

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        logger.log(f'saving at step {self.update_steps} ...')
        dir_name = f'{self.save_dir}/model-{self.update_steps//1000:03d}k'
        self.model.module.save_pretrained(dir_name)
        self.dataset.dataset.tokenizer.save_pretrained(dir_name)
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'update_steps': self.update_steps,
            'iter_steps': self.iter_steps,
            'epoch': self.epoch,
            'epoch_length': self.epoch_length,
        }, dir_name + '/status.pt')
        logger.log(
            f'saved at {dir_name}')
        if os.path.exists(f'{self.save_dir}/model-last'):
            os.remove(f'{self.save_dir}/model-last')
        os.symlink(dir_name, f'{self.save_dir}/model-last')

    def load_status(self, dir_name):
        """won't load model parameter"""
        status = torch.load('/'.join([dir_name, 'status.pt']))
        self.scaler.load_state_dict(status['scaler'])
        self.optimizer.load_state_dict(status['optimizer'])
        self.scheduler.load_state_dict(status['lr_scheduler'])
        self.update_steps = status['update_steps']

        if self.epoch_length == status['epoch_length']:  # reloadable
            self.epoch = status['epoch']
            self.iter_steps = status['iter_steps']

    def log(self, stats):
        logging_str = f'u: {self.update_steps}'
        logging_str += f'\tep: {self.epoch}'
        logging_str += f'\t({self.in_epoch_step}/{self.epoch_length})'
        for k, v in stats.items():
            format_str = f'\t{k}: {v}'
            if isinstance(v, float):
                format_str = f'\t{k}: {v:.4g}'
            logging_str += format_str
        logger.log(logging_str)


def distributed_main(rank, args, model, dataset, tokenizer):
    dist.init_process_group(backend='nccl', rank=rank, world_size=args.world_size)

    device = torch.device(rank)
    torch.cuda.set_device(device)
    logger.log(f'GPU-{rank} process started')

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if args.model_name is not None:
        if os.path.exists('/'.join([args.model_name, 'pytorch_model.bin'])):
            model = model.from_pretrained(args.model_name)
            logger.log(f'loading weight from {args.model_name}')
        else:
            logger.log(f'{args.model_name} does not exist, using random init')
    model = model.to(device)
    
    dataiter = DataCollector(
        dataset=dataset,
        num_workers=args.num_workers // args.world_size,
        rank=rank,
        world_size=args.world_size,
        shuffle=True,
    )
    if dist.get_rank() == 0:
        logger.log('Building trainer...')
    trainer = Trainer(
        args=args,
        rank=rank,
        dataset=dataiter,
        model=model,
        optimizer_name=args.optim,
        scheduler_name=args.scheduler,
        criterion_name=args.criterion,
        optimizer_params={'lr': args.learning_rate},
        scheduler_params={'num_warmup_steps': args.num_warmup_steps,
                          'num_training_steps': args.num_training_steps},
        criterion_params={
            'ignore_index': model.config.pad_token_id,
            'temperature': args.temperature,
            'label_smoothing': args.label_smoothing,
            'reduction': 'none'},
        update_per_save=args.update_per_save,
        update_per_log=args.update_per_log,
        iter_per_update=args.iter_per_update,
        num_training_steps=args.num_training_steps,
        save_dir=args.save_dir,
        fp32=args.fp32
    )
    if dist.get_rank() == 0:
        logger.log('Training started')
    trainer.train()
    
    exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cn', '--config-name', required=True)
    parser.add_argument('-mn', '--model-name')
    parser.add_argument('-tn', '--tokenizer-name')
    parser.add_argument('-d', '--data', required=True)
    parser.add_argument('-s', '--suffixes', nargs=2, required=True)
    parser.add_argument('--max-tokens', type=int, required=True)
    parser.add_argument('--optim', default='adam')
    parser.add_argument('--criterion', default='temperature_cross_entropy')
    parser.add_argument('--scheduler', default='linear')
    parser.add_argument('-lr', '--learning-rate', type=float, required=True)
    parser.add_argument('--num-training-steps', type=int, required=True)
    parser.add_argument('--num-warmup-steps', type=int, required=True)
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--update-per-log', type=int, default=100)
    parser.add_argument('--iter-per-update', type=int, default=1)
    parser.add_argument('--update-per-save', type=int, default=1000)
    parser.add_argument('--fp32', action="store_true")
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--length-ratio', type=float, default=float('inf'))
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--world-size', default=None)
    parser.add_argument('--label-smoothing', type=float, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-norm', type=float, default=1.0)

    args = parser.parse_args()

    if args.model_name is None:
        setattr(args, 'model_name', os.path.join(args.save_dir, "model-last"))
    if args.tokenizer_name is None:
        setattr(args, 'tokenizer_name', args.config_name)
    if args.world_size is None:
        setattr(args, 'world_size', torch.cuda.device_count())
    if args.num_workers is None:
        env_cpus = int(os.environ.get('SLURM_CPUS_ON_NODE', mp.cpu_count()))
        setattr(args, 'num_workers', max(env_cpus, args.world_size))
    logger.log(args)

    logger.log('Building tokenizer')
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_name)
    logger.log('Building dataset...')
    dataset_sent = DatasetForSeq2Seq(args.data, 'train', args.suffixes,
                                       args.max_length, args.length_ratio, tokenizer, args.num_workers)
    logger.log('Bachifying data...')
    dataset = BatchedDataset(dataset_sent, args.max_tokens, tokenizer, args.num_workers)
    logger.log('Building model...')
    config = transformers.AutoConfig.from_pretrained(args.config_name)
    model = transformers.AutoModelForSeq2SeqLM.from_config(config)
    logger.log(model)

    mp.spawn(
        distributed_main,
        nprocs=args.world_size,
        args=(args, model, dataset, tokenizer),
        join=True,
    )
