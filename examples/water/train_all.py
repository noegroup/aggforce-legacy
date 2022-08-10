
import os


for rigid in [True, False]:
    for force_map in ["slice", "agg"]:
        for seed in range(5):
            for train_percentage in [1, 5, 10, 25, 50, 75, 100]:
                fraction =  train_percentage / 100 * 0.8 
                command = (
                    f"python train.py --train-fraction {train_fraction:.2} "
                    f"--force-map {force_map} --random-seed {seed} "
                    f"--log_every_n_steps 5 --batch-size 128"
                )
                if rigid:
                    command += " --rigid"
                else:
                    command += " --no-rigid"
                os.system(command)
