
import pandas as pd


with open('benchmark_log.csv') as file:
    set_data = pd.read_csv(file)

with open('benchmark_log_random.csv') as file:
    random_data = pd.read_csv(file)


unique_items = random_data['object'].unique()

with open('output.txt', 'w') as output_file:
    for item in unique_items:

        extract_set = set_data.loc[set_data['object'] == item]
        extract_random = random_data.loc[random_data['object'] == item]

        set_success = extract_set['success'].value_counts()
        random_success = extract_random['success'].value_counts()

        try: set_ones = set_success[1]
        except: set_ones = 0

        try: set_zeros = set_success[0]
        except: set_zeros = 0

        try: random_ones = random_success[1]
        except: random_ones = 0

        try: random_zeros = random_success[0]
        except: random_zeros = 0

        if set_ones != 0: set_rate = set_ones / (set_ones + set_zeros)
        else: set_rate = 0

        if random_ones != 0: random_rate = random_ones / (random_ones + random_zeros)
        else: random_rate = 0

        output_file.write(f'\nItem name: {item}\n')
        output_file.write(f'Algorithm success rate: {set_rate}\n')
        output_file.write(f'Random points success rate: {random_rate}\n')
        output_file.write(f'Sample sizes: {set_ones + set_zeros} | {random_ones + random_zeros}\n')