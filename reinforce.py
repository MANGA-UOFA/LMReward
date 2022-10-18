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
import copy
import time


def shift_left(tensor, fill_token_id):
    assert len(tensor.shape) == 2
    start_ids = torch.ones((tensor.shape[0], 1)).to(
        tensor) * fill_token_id
    return torch.cat([tensor[..., 1:]], start_ids, dim=-1)

def shift_right(tensor, fill_token_id):
    assert len(tensor.shape) == 2
    start_ids = torch.ones((tensor.shape[0], 1)).to(
        tensor) * fill_token_id
    return torch.cat([start_ids, tensor[..., :-1]], dim=-1)


def reward_fn(reward_model, input_ids, output_ids, args, pad_token_id=None):
    pad_token_id = pad_token_id or reward_model.config.pad_token_id

    reward_model.eval()
    input_ids = input_ids.repeat(1, 1)
    model_inputs = {
        'input_ids': input_ids,
        'attention_mask': input_ids != pad_token_id
    }

    start_tokens = torch.zeros((output_ids.shape[0], 1)).to(output_ids)
    start_tokens.fill_(reward_model.config.decoder_start_token_id)
    
    model_inputs['decoder_input_ids'] = torch.cat([start_tokens, output_ids[..., :-1]], dim=-1)
    model_inputs['decoder_attention_mask'] = output_ids != pad_token_id
    model_inputs['use_cache'] = False

    outputs = reward_model(**model_inputs)
    
    logits = outputs.logits # (B, L, V)
    logits = logits - torch.mean(logits, dim=-1, keepdim=True)

    selection_value = torch.gather(logits, -1, output_ids[..., None]).squeeze(-1)
    selection_value.masked_fill_(output_ids == pad_token_id, 0.0)

    next_logits = torch.roll(logits, -1, 1)
    if args.softmax:
        next_state_value = torch.sum(torch.softmax(next_logits, dim=-1) * next_logits, dim=-1)
    else:
        next_state_value = next_logits.max(dim=-1)[0] # (B, L)

    next_state_value.masked_fill_(output_ids == pad_token_id, 0.0)
    next_state_value.masked_fill_(output_ids == reward_model.config.eos_token_id, 0.0)

    if args.baseline:
        baselines = torch.sum(torch.softmax(logits, dim=-1) * logits, dim=-1)
        baselines.masked_fill_(output_ids == pad_token_id, 0.0)
    else:
        baselines = None
    
    return selection_value - next_state_value, baselines


logger = Logger()


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
            if dist.get_rank() == 0:
                logger.log('grad nan/inf detected')
            return
        return optimizer.step()

    def update(self):
        return

    def get_scale(self):
        return 1.0

    def unscale_(self, optimizer):
        return


def actor_proc(queue, model, dataset, args, world_size, local_rank, device, free_to_go):
    dataset = DataCollector(
        dataset=dataset,
        num_workers=args.num_workers // world_size,
        rank=local_rank,
        world_size=world_size,
        shuffle=True,
    )
    logger.log('proc started')
    config = model.config
    amp_context = nullcontext() if args.fp32 else torch.cuda.amp.autocast()
    epoch_length = len(dataset)
    queue.put(epoch_length)

    while free_to_go.value == 0:
        time.sleep(0.1)

    with torch.no_grad(), amp_context:
        while True:
            dataiter = iter(dataset)
            for _ in range(epoch_length):
                data = next(dataiter)
                model.eval()
                enc_input = data['src']['input_ids'].to(device)
                
                model_kwargs = dict()
                model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
                    enc_input, model_kwargs)
                # set input_ids as decoder_input_ids
                if "decoder_input_ids" in model_kwargs:
                    input_ids = model_kwargs.pop("decoder_input_ids")
                else:
                    input_ids = model._prepare_decoder_input_ids_for_generation(
                        enc_input.shape[0]
                    )
                input_ids, model_kwargs = model._expand_inputs_for_generation(
                    input_ids,
                    is_encoder_decoder=config.is_encoder_decoder, **model_kwargs
                )
                ready_mask = torch.zeros((enc_input.shape[0], 1)).bool().to(device=device)
                
                max_length = max(
                    args.min_max_length,
                    min(
                        args.max_length,
                        int(args.length_ratio * args.max_tokens - input_ids.numel()) // input_ids.shape[0]
                    )
                )
                b_probs = []
                for l in range(max_length):
                    model_inputs = model.prepare_inputs_for_generation(
                        input_ids, **model_kwargs)
                    outputs = model(**model_inputs,
                                    return_dict=True,
                                    output_hidden_states=config.output_hidden_states,
                                    output_attentions=config.output_attentions)
                    
                    logits = outputs.logits[:, -1, :] / args.temperature # (beam, vocab)

                    def top_k_mask_out(t, k, flat=False):
                        _list = torch.topk(t, k, -1)[0]
                        lb = _list[..., -1, None]
                        # index = torch.argsort(t, dim=-1, descending=True)[..., _rep+k:]
                        # return torch.scatter(t, -1, index, -torch.inf)
                        if flat:
                            return torch.where(t >= lb, 0.0, -torch.inf)
                        return torch.masked_fill(t, t < lb, -torch.inf)
                    if args.topk is not None:
                        logits = top_k_mask_out(logits, args.topk, flat=(l==0) and args.init_flat)
                    prob = torch.softmax(logits, -1) # (B, V)

                    next_tokens = torch.multinomial(prob, 1) # (B, 1)
                    b_probs.append(prob.gather(-1, next_tokens))

                    next_tokens.masked_fill_(ready_mask, config.pad_token_id)
                    eos_mask = next_tokens == config.eos_token_id
                    
                    input_ids = torch.cat([input_ids, next_tokens], dim=-1)
                    model_kwargs = model._update_model_kwargs_for_generation(
                        outputs, model_kwargs, is_encoder_decoder=config.is_encoder_decoder
                    )

                    ready_mask = eos_mask | ready_mask
                    if ready_mask.all():
                        break
                
                b_probs = torch.cat(b_probs, dim=-1)
                input_ids = input_ids[..., 1:]

                ready_index = torch.arange(ready_mask.shape[0], device=ready_mask.device).masked_select(ready_mask.squeeze())
                enc_input = enc_input.index_select(0, ready_index)
                b_probs = b_probs.index_select(0, ready_index)
                input_ids = input_ids.index_select(0, ready_index)
                
                queue.put((enc_input, input_ids, b_probs, ready_mask))
                del data
            del dataiter


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
        tokenizer,
    ):
        assert optimizer_name in ['adam']
        assert criterion_name in ['cross_entropy']

        device = torch.device(dist.get_rank())
        
        if optimizer_name == 'adam':
            optimizer_cls = torch.optim.Adam

        if criterion_name == 'cross_entropy':
            criterion_cls = torch.nn.CrossEntropyLoss

        self.args = args
        self.rank = rank
        self.config = model.config
        self.gen_model = copy.deepcopy(model)
        self.model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device],
            output_device=device
        )
        self.dataset = dataset
        self.optimizer = optimizer_cls(
            self.model.parameters(), **optimizer_params)
        self.scheduler = transformers.get_scheduler(name=scheduler_name, optimizer=self.optimizer, **scheduler_params)
        self.criterion = criterion_cls(**criterion_params)
        self.tokenizer = tokenizer

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

        self.reward_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.reward_model).to(device)
        self.reward_model.requires_grad_(False)

        num_actor = 2
        self.queue = mp.Queue(maxsize=2)
        self.actor_procs = []
        free_to_go = mp.Value('i', 0)
        
        for i in range(num_actor):
            self.actor_procs.append(
                mp.Process(
                    target=actor_proc,
                    args=(self.queue, self.gen_model , self.dataset, args, args.world_size * num_actor, self.rank * num_actor + i, device, free_to_go)))
        for p in self.actor_procs:
            p.start()

        epoch_lengths_per_gpu = [None] * dist.get_world_size()
        dist.all_gather_object(epoch_lengths_per_gpu, sum(self.queue.get() for _ in range(num_actor)))
        self.epoch_length = min(epoch_lengths_per_gpu) * self.args.repeat

        free_to_go.value = 1

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
        config = self.model.module.config
        factory = self.model.module
        os.makedirs(self.save_dir, exist_ok=True)

        
        while self.update_steps < self.num_training_steps:
            for in_epoch_step in range(self.epoch_length):
                self.in_epoch_step = in_epoch_step + 1

                if self.update_steps >= self.num_training_steps:
                    break

                data = self.queue.get()

                if self.iter_steps % self.epoch_length != in_epoch_step:  # reloaded:
                    del data
                    continue

                ddp_context = nullcontext() if (self.iter_steps + 1) % self.iter_per_update == 0 else self.model.no_sync()
                amp_context = nullcontext() if self.fp32 else torch.cuda.amp.autocast()
                
                enc_input, input_ids, b_probs, ready_mask = data

                dec_input = shift_right(input_ids, config.decoder_start_token_id)
                attention_mask = (input_ids != config.pad_token_id)
                
                model_inputs = {
                    'input_ids': enc_input,
                    'attention_mask': enc_input != config.pad_token_id,
                    'decoder_input_ids': dec_input,
                    'decoder_attention_mask': attention_mask,
                    'return_dict': True,
                    'use_cache': False,
                }
                    
                with ddp_context:
                    if enc_input.shape[0] > 0:
                        with amp_context:
                            self.model.train()
                            output = self.model(**model_inputs)
                            logits = output.logits # (BATCH, LENGTH, VOCAB)
                            probs = torch.softmax(logits, dim=-1).masked_fill(attention_mask.logical_not()[..., None], 0.0)
                            with torch.no_grad():
                                rewards, baselines = reward_fn(
                                    self.reward_model,
                                    enc_input,
                                    input_ids,
                                    self.args,
                                    config.pad_token_id
                                ) # (BATCH, LENGTH)
                                rewards = rewards.flip([-1])
                                rewards = torch.cumsum(rewards, dim=-1)
                                rewards = rewards.flip([-1])
                                
                                if baselines is not None:
                                    rewards = rewards - baselines
                                rewards = rewards / self.args.denom
                                if self.args.sent_level:
                                    rewards = rewards[..., :1].repeat(1, rewards.shape[-1]) # (B, L)
                                rho = torch.gather(probs, -1, input_ids[..., None]).squeeze(-1)
                                rho = rho / b_probs
                                rho = rho.masked_fill(attention_mask.logical_not(), 0.0)
                                rewards = rewards * rho
                                if self.args.reward_clip is not None:
                                    rewards = torch.clip(rewards, min=-self.args.reward_clip, max=self.args.reward_clip)
                                
                                rewards = rewards.masked_fill(attention_mask.logical_not(), 0.0)
                            loss = 0
                            lprobs = torch.log_softmax(logits, dim=-1)
                            if self.args.entropy != 0:
                                entropy = torch.mean(-lprobs, -1)
                                entropy = torch.masked_fill(entropy, attention_mask.logical_not(), 0.0)
                                entropy = entropy.sum() / attention_mask.sum()

                                loss = loss + entropy * self.args.entropy
                            lprobs = torch.gather(lprobs, -1, input_ids[..., None]).squeeze(-1) # (B, L)
                            lprobs = torch.masked_fill(lprobs, attention_mask.logical_not(), 0.0)
                            loss = loss + (-lprobs * rewards).sum() / attention_mask.sum()
                            loss_avg.take(loss.item())
                            loss = loss / self.iter_per_update # gradient accumulation
                            loss = loss * ready_mask.sum() / ready_mask.numel()
                    else:
                        loss = torch.zeros(1, requires_grad=True, device=ready_mask.device)
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.log({'warning': 'forward nan/inf detected'})
                        loss = loss.fill_(0.0)
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
                    if self.update_steps % self.args.update_per_sync == 0:
                        self.gen_model.load_state_dict(self.model.module.state_dict())
                    if dist.get_rank() == 0:
                        try:
                            if self.update_steps % self.update_per_log == 0:
                                self.log({
                                    'lr': self.scheduler.get_last_lr()[0],
                                    'loss': loss_avg.pop(),
                                    'scaling': self.scaler.get_scale(),
                                    'grad_norm': grad_norm_avg.pop(),
                                    'entropy': entropy.item() if self.args.entropy != 0.0 else None,
                                    'valid_rate': ready_mask.sum() / ready_mask.numel(),
                                })
                        except Exception as inst:
                            print(inst)
                        if self.update_steps % self.update_per_save == 0:
                            self.save()
                    
                del data
            self.epoch += 1

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        logger.log(f'saving at step {self.update_steps} ...')
        dir_name = f'{self.save_dir}/model-{self.update_steps//1000:03d}k'
        self.model.module.save_pretrained(dir_name)
        self.dataset.tokenizer.save_pretrained(dir_name)
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

    if model.config.pad_token_id == model.config.decoder_start_token_id:
        setattr(model.config, 'pad_token_id', tokenizer.vocab['<extra_id_1>'])

    if dist.get_rank() == 0:
        logger.log('Building trainer...')
    trainer = Trainer(
        args=args,
        rank=rank,
        dataset=dataset,
        model=model,
        optimizer_name=args.optim,
        scheduler_name=args.scheduler,
        criterion_name=args.criterion,
        optimizer_params={'lr': args.learning_rate},
        scheduler_params={'num_warmup_steps': args.num_warmup_steps,
                          'num_training_steps': args.num_training_steps},
        criterion_params={
            'ignore_index': model.config.pad_token_id,
            'label_smoothing': args.label_smoothing,
            'reduction': 'none'},
        update_per_save=args.update_per_save,
        update_per_log=args.update_per_log,
        iter_per_update=args.iter_per_update,
        num_training_steps=args.num_training_steps,
        save_dir=args.save_dir,
        fp32=args.fp32,
        tokenizer=tokenizer,
    )
    if dist.get_rank() == 0:
        logger.log('Training started')
    trainer.train()


if __name__ == "__main__":
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('-cn', '--config-name', required=True)
    parser.add_argument('-mn', '--model-name')
    parser.add_argument('-tn', '--tokenizer-name')
    parser.add_argument('-d', '--data', required=True)
    parser.add_argument('-s', '--suffixes', nargs=2, required=True)
    parser.add_argument('--max-tokens', type=int, required=True)
    parser.add_argument('--optim', default='adam')
    parser.add_argument('--criterion', default='cross_entropy')
    parser.add_argument('--scheduler', default='linear')
    parser.add_argument('-lr', '--learning-rate', type=float, required=True)
    parser.add_argument('--num-training-steps', type=int, required=True)
    parser.add_argument('--num-warmup-steps', type=int, required=True)
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--update-per-log', type=int, default=10)
    parser.add_argument('--iter-per-update', type=int, default=1)
    parser.add_argument('--update-per-save', type=int, default=1000)
    parser.add_argument('--fp32', action="store_true")
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--length-ratio', type=float, default=2)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--world-size', type=int, default=None)
    parser.add_argument('--label-smoothing', type=float, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-norm', type=float, default=1.0)
    parser.add_argument('--reward-model', required=True)
    parser.add_argument('--reward-clip', type=int)
    parser.add_argument('--denom', type=float, default=1.0)
    parser.add_argument('--topk', type=int) 
    parser.add_argument('--softmax', action="store_true")
    parser.add_argument('--baseline', action="store_true")
    parser.add_argument('--entropy', type=float, default=0.0)
    parser.add_argument('--min-max-length', type=int, default=8)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--update-per-sync', type=int, default=5000)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--init-flat', action='store_true')
    parser.add_argument('--sent-level', action='store_true')
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

    distributed_procs = []
    for i in range(args.world_size - 1):
        proc = mp.Process(
            target=distributed_main,
            args=(i, args, model, dataset, tokenizer)
        )
        proc.start()
        distributed_procs.append(proc)
    distributed_main(args.world_size - 1, args, model, dataset, tokenizer)
    for p in distributed_procs:
        p.join()
