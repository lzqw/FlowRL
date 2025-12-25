from utilis.config import Config


default_config = Config({
    "seed": 0,
    "tag": "default",
    "start_steps": 5000,
    "cuda": True,
    "num_steps": 1000001,
    "save": True,
    
    "env_name": "Walker2d-v4",  
    "eval": True,
    
    "eval_numsteps": 10000,
    "eval_times": 5,
    "replay_size": 1000000,

    "algo": "FlowAC",
    "policy": "Flow", 
    "steps": 10,
    "gamma": 0.99, 
    "tau": 0.95,
    "lr": 0.0003, #0.0003
    "batch_size": 256, 
    "updates_per_step": 1,
    "target_update_interval": 2, # for delayed policy update and target network update
    "hidden_size": 256,

    "quantile": 0.9,  # default ,but needed to be tuned for unseen tasks ,in unseen tasks for humanoid_bench ,we recommened to set quantile= 0.8
    "epochs":50 ,
    "eps" : 0.1 
})
