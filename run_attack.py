import os

# Define the values of eps

eps_values = [0.001, 0.01]
num_steps_values = [10, 20, 30]

# Loop through each value and execute the script
for eps in eps_values:
    for num_steps in num_steps_values:
        print(f"Running evaluate.py with eps={eps}, num_steps={num_steps}")
        command = f"python evaluate.py --path cnn --model cnn --attack pgd --epsilon {eps} --num_steps {num_steps}"
        os.system(command)