import pandas as pd

df = pd.read_csv("artifacts/test_interpolated.csv")

car_ids = df['car_id'].values

count_dict = {}
counter = 1
result = []
for num in car_ids:
    if num not in count_dict:
        count_dict[num] = counter
        counter += 1
    result.append(count_dict[num])

print(len(result))
print(car_ids.shape)
