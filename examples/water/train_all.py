
import os

SUBMIT = True
DRY_RUN = False #True

SEEDS = range(1, 11)
PERCENTAGES = [0.1, 0.5, 1, 5, 10, 50, 100]
#PERCENTAGES = [50, 100]

for train_percentage in PERCENTAGES:
    for seed in SEEDS:
        for rigid in [True, False]:
            for force_map in ["slice", "agg"]:
            
                train_fraction =  train_percentage / 100 * 0.8 
                command = (
                    f"train.py --train-fraction {train_fraction:.4f} "
                    f"--force-map {force_map} --random-seed {seed} "
                    f"--log_every_n_steps 1 "
                    f"--n-rbf 200 --max_epochs 1000"
                )
                if rigid:
                    command += " --rigid"
                else:
                    command += " --no-rigid"

                if SUBMIT:
                    pre_command = f"sbatch -J rig{rigid}_{force_map}_{seed}_{train_fraction:.4f} -o slurm_out/%j.o "
                else:
                    pre_command = "python "
                command = pre_command + command

                
                if DRY_RUN:
                    print(command)
                else:
                    os.system(command)
