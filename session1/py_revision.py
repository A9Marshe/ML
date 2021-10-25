import numpy as np

# Define 1D - 2D arrays 
ar_1D = np.array([1, 2, 3, 4, 5])
ar_2D = np.array([[1, 2, 3], [4, 5, 6],[7, 8, 9] ])

#Number of Dimensions
print("Dim",ar_1D.ndim) ,print ("size",np.size(ar_1D ))  
print("Dim",ar_2D.ndim) ,print ("size",np.size(ar_2D )) ## By default, give the total number of elements.

print(np.size(ar_2D, 0))     # number of rows
print(np.size(ar_2D, 1))     # number of columns
print('max= ',np.max(ar_2D)) #max of all elements
print('min= ',np.min(ar_2D))
print('sum= ',np.sum(ar_2D)) # sum of array

# extract first column ( indexes starts with 0)
fc = ar_2D[:, 0]
print('First column',fc)

# create 3*1 array
x0 = np.ones((3, 1),'i')

# add it as first colmn to previous array 
x = (np.column_stack((x0, ar_2D))) # or x=(np.column_stack(( ar_2D , x0)))
 
print(x)
print(x.reshape(2,6))



# sqr of array elements
print('square =',np.square(ar_2D))

# create range and vonvert it to array
print(np.mat(range(0, 10)))  # we can use :print(np.mat(range(0, 10)).reshape(5,2))

# martix multiplication
a = np.array([[1 ],[2],[3],[4] ] )
b = np.array([[1,2]])
print (np.matmul(a,b))

#Linear algebra
x = np.array([[1,2],[3,4]]) 
y = np.linalg.inv(x) 
print (x)
print ('linalg.inv of x=',y)
print(np.matmul(x,y))

#mean , std , where   , squeeze , abs , dot, log
