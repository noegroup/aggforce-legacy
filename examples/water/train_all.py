
import os


for rigid in [True, False]:
    for force_map in ["slice", "agg"]:
        for seed in range(5):
            for train_fraction in [0.04, 0.08, 0.4, 0.8]:
                command = (
                    f"python train.py --train-fraction {train_fraction} "
                    f"--force-map {force_map} --random-seed {seed} "
                    f"--rigid {rigid} "
                    f"--log_every_n_steps 5 --batch-size 128"
                )
                os.system(command)
