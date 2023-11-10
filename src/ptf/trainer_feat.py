def build_truncated_shuffled_dataloader(
    dataset: Dataset, 
    batch_size: int, 
    gradient_accumulation_step: int = 1, 
    generator: torch.Generator | None = None
) -> DataLoader:
    new_batch_size: int = batch_size // gradient_accumulation_step
    new_num_samples: int = len(dataset) // batch_size * batch_size
    
    sampler = RandomSampler(
        dataset, num_samples=new_num_samples, generator=generator
    )
    return DataLoader(dataset, new_batch_size, sampler=sampler)


class Trainer:
    r""" Base class for all trainers
    
    """
    def __init__(
        self,
        model, 
        *,
        epoch_num: int,
        batch_size: int,
        gradient_accumulation_step: int = 1,
        seed: int | None = None,
        device: str | torch.device | None = "cpu"
    ) -> None:
        self.model = model 
        
        self.start_epoch: int = 0
        self._num_train_epochs: int = epoch_num
        self.batch_size: int = batch_size
        self.gradient_accumulation_steo: int = gradient_accumulation_step
        
        if not isinstance(seed, int):
            seed = random.randint(- 2 ** 63, 2 ** 64 - 1)
        self.init_seed = seed
        self.data_rng = torch.manual_seed(seed)
        
        self.device = torch.device(device)
        
        self.network_weights_not_initiated: bool = True
    
    @property
    def num_train_epochs(self):
        return self._num_train_epochs
    
    @property
    def epoch(self):
        return self.start_epoch + getattr(self, "local_epoch", 0)
    
    def loop(self, dataset, network, optimizer):
        #? Before Loop
        #?  - Load checkpoint state 加载检查点状态
        #?  - Initiate network weights 初始化权重
        #?  - Initiate data loader 初始化数据加载器
        #?  - Send network to proper device 将模型转移到合适的设备
        
        # NOTE 
        checkpoint_file: Path = None
        pretrained_weights_file: Path = None
        optimizer_state_file: Path = None
        convert: Callable[Batch, Batch | CudaBatch] = None
        
        if checkpoint_file and checkpoint_file.exists():
            ckpt_state_dict = torch.load(checkpoint_file, map_location="cpu")
            self.start_epoch = ckpt_state_dict["epoch"]
            self.init_seed = ckpt_state_dict["init_seed"]
            self.data_rng.set_state(ckpt_state_dict["data_rng_state"])
        
        if pretrained_weights_file and pretrained_weights_file.exists():
            network_state_dict = torch.load(pretrained_weights_file, map_location="cpu")
            network.load_state_dict(network_state_dict)
            self.network_weights_not_initiated = False
        
        if optimizer_state_file and optimizer_state_file.exists():
            optimizer_state_dict = torch.load(optimizer_state_file, map_location="cpu")
            optimizer.load_state_dict(optimizer_state_dict)
        
        if self.network_weights_not_initiated:
            network.init_weights(self.init_seed)
        
        network.to(self.device)
        
        dataloader = build_truncated_shuffled_dataloader(
            dataset=dataset, batch_size=self.batch_size, 
            gradient_accumulation_step=self.gradient_accumulation_steo,
            generator=self.data_rng
        )
        
        self.local_epoch = 0
        for local_epoch_index in range(self.num_train_epochs):
            self.local_epoch += 1
            network.train()
            
            for local_step_index, batch in enumerate(dataloader):
                
                batch = convert(batch)
                loss_dict = self.model.compute_losses(network, batch)
                total_loss = self.model.total_loss(loss_dict)
                
                if 
            
        
        