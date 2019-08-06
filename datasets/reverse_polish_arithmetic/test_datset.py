from arithmetic_dataset import Dataset

data = Dataset('fixed_L2_1e2/training.txt','fixed_L2_1e2/testing.txt')

input, output, line = data.training_item(0)

print('LINE',line)
print('INPUT',input)
print('OUTPUT',output)
