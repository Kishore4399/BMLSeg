import os
import warnings
import torch

import wandb

import sys
sys.path.append('lib/')

from lib.utils import set_seed, dist_setup, get_conf
import lib.trainers as trainers



def main():

    args = get_conf()

    args.test = False
    set_seed(args.seed)
    main_worker(args.gpu, args)

def main_worker(gpu, args):
    args.gpu = gpu
    dist_setup(args)

    trainer_class = getattr(trainers, f'{args.trainer_name}', None)
    assert trainer_class is not None, f"Trainer class {args.trainer_name} is not defined"
    trainer = trainer_class(args)

    if args.rank == 0 and not args.disable_wandb:
        if args.wandb_id is None:
            args.wandb_id = wandb.util.generate_id()

        run = wandb.init(project=f"{args.proj_name}_{args.dataset}", 
                        name=args.run_name, 
                        config=vars(args),
                        id=args.wandb_id,
                        resume='allow',
                        dir=args.output_dir)

    # create model
    trainer.build_model()
    # create optimizer
    trainer.build_optimizer()
    # resume training
    if args.resume:
        trainer.resume()
    trainer.build_dataloader()

    trainer.run()

    if args.rank == 0 and not args.disable_wandb:
        run.finish()

if __name__ == '__main__':
    main()



