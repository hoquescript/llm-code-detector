import pandas as pd;
import numpy as np;
import time


# # Regular Array
# data = [1, 2, 3]
# result = data * 2
# print(result)

# # Numpy Array
# data = np.array([1,2,3])
# result = data*2
# print(result)



# regular_list = range(10_000_000)
# start = time.time()
# multiply_list = [x * 2 for x in regular_list]
# end = time.time()
# print(f"Python List Time: {end - start:.4f} seconds")

# numpy_list = np.array(range(10_000_000))
# start = time.time()
# multiply_list = numpy_list * 2
# end = time.time()
# print(f"Numpy List Time: {end - start:.4f} seconds")


import numpy as np

# A list of lists becomes a 2D Matrix
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print("The Matrix:\n", matrix)
print("\nShape:", matrix.shape)
print("Data Type:", matrix.dtype)