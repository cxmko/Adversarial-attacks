import os

# Define the values of eps
## To do 14
eps_values = [0.1, 0.01, 0.001]

# Loop through each value and execute the script
for eps in eps_values:
    print(f"Running evaluate.py with eps={eps}")
    command = f"python evaluate.py --path cnn --model cnn --attack fgsm --eps {eps}"
    os.system(command)


