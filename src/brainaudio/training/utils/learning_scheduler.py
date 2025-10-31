import math 
from torch.optim.lr_scheduler import LambdaLR

def create_learning_rate_scheduler(args, optim):
    

    scheduler_type = args['learning_scheduler']
    
    learning_rate = args['learning_rate']
    
    # cosine learning rate hparams
    learning_rate_min = args['learning_rate_min']
    learning_rate_decay_steps = args['learning_rate_decay_steps']
    learning_rate_warmup_steps = args['learning_rate_warmup_steps']
    
    # multistep learning rate hparams 
    gamma = args['gamma']
    milestones = args['milestones']

    def learning_rate_lambda(current_step, min_learning_rate_ratio, decay_steps, warmup_steps):
        
        '''
        Create lr lambdas for each param group that implement cosine decay
        Different lr lambda decaying for day params vs rest of the model
        
        current_step: current batch index
        min_learning_rate_ratio: Learning rate shou
        '''
        
        
        # Warmup phase
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # Cosine decay phase
        if scheduler_type == 'cosine':
            
            if current_step < decay_steps:
                progress = float(current_step - warmup_steps) / float(
                    max(1, decay_steps - warmup_steps)
                )
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                # Scale from 1.0 to min_learning_rate_ratio
                return max(min_learning_rate_ratio, min_learning_rate_ratio + (1 - min_learning_rate_ratio) * cosine_decay)
            
            # After cosine decay is complete, maintain min_learning_rate_ratio
            return min_learning_rate_ratio
        
        if scheduler_type == 'multistep':
            
            power = sum([1 for m in milestones if current_step >= m])
            return gamma ** power
        
        if scheduler_type == 'None':
            
            return 1 
            
        
    return LambdaLR(optim,  lambda step: learning_rate_lambda(step, learning_rate_min / learning_rate, learning_rate_decay_steps, learning_rate_warmup_steps), -1)
    
    
    