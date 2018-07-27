import numpy as np

data = [[1,2,3.23], [4,5,6], [2,2,3]]
print(data)
arr1 = np.array(data)
print(data)
arr2 = arr1*2
print(arr2)
print(arr1 + arr2)
print(arr1.ndim)

print("arr1.shape:\n", arr1.shape)
arr1.astype(np.int32)
print("arr1:\n", arr1)
print("arr1.dtype:", arr1.dtype)