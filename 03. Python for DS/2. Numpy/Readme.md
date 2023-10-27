<!-- 
    Author : Kannan Jayachandran
    File : Readme.md (Numpy documentation)
    Not reviewed, Not completed
 -->

<h1 align="center"> Numpy </h1>

`Numpy` or `Numerical Python` is a python library for scientific and numerical applications. It is the preferred tool for linear algebra operations and working with multi-dimensional array objects. It is optimized for `vectorized operations` and thus it is extensively used for performing operations on individual elements. The primary data structure in Numpy is the `ndarray` (n-dimensional array). It is a `homogeneous multidimensional` array of fixed-size items. 

### Some basic concepts of numpy are:

- `axes` : Dimensions in numpy are called _axes_. In numpy, the axes are defined for arrays with more than one dimension

- `rank` : Number of axes

- `ndarray.shape` - returns a tuple consisting of array dimensions (similar to (rows, columns))

- `ndarray.ndim` - returns the number of array dimensions

- `ndarray.size` - returns the total number of elements in the array

- `ndarray.dtype` - returns the type of elements in the array

- `ndarray.itemsize` - Attribute of n-darray, returns the size of each element in the array in bytes

- `ndarray.data` - returns the buffer containing the actual elements of the array

### Creating arrays

- `np.array()` - Creates an array or converts a list into a numpy array

- `np.arange()` - Creates a range of numbers as a numpy array. It can also take float arguments unlike `range`

- `np.linspace(start, stop, number of elements)` - Creates an array of evenly spaced numbers over a specified interval. We use it when we want to divide a range into equal parts.

- `np.empty(shape)` - Creates an array of the specified shape with random values. The values are not necessarily zero. It is faster than `np.zeros` as it does not initialize the array.

- `np.zeros(shape)` - Creates an array of zeros of the specified shape. The default _dtype_ is float64.

- `np.ones(shape)` - Creates an array of ones of the specified shape. The default _dtype_ is float64.

- `np.full(shape, fill_value)` - Creates an array of the specified shape with the specified fill value. The default _dtype_ is float64. 

### Stacking arrays

- `np.vstack()` - Stacks arrays vertically. In order to stack arrays vertically, the number of columns in the arrays must be the same.

- `np.hstack()` - Stacks arrays horizontally. In order to stack arrays horizontally, the number of rows in the arrays must be the same.

- `np.dstack()` - Stacks arrays depth wise. In order to stack arrays depth wise, the number of rows and columns in the arrays must be the same.

- `np.concatenate()` - Concatenates arrays along a specified axis. The default axis is 0, which is the row axis.


### Indexing - Slicing - Reshaping

- `np.array[index]` - Indexing an array. It returns a copy of the array.

- `np.array[0, 3]` - Indexing a 2D array at row 0 and column 3. It returns a copy of the array.

- `np.array[0:3]` - Slicing an array. It returns a copy of the array.

- `np.reshape()` - Reshapes an array into a new shape. The number of elements in the array should be the same as the number of elements in the new shape. It returns a new array.

- `np.resize()` - Resizes an array into a new shape. The number of elements in the array can be different from the number of elements in the new shape, if the number of elements is less than the number of elements in the new shape then the array will be filled by repeating the elements. It returns a new array.

- `np.flatten()` - Flattens an array into a 1D array. It returns a copy of the array.

- `np.ravel()` - Flattens an array into a 1D array. It performs the operation in place.

> <picture>
>   <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Mqxx/GitHub-Markdown/main/blockquotes/badge/dark-theme/warning.svg">
>   <img alt="Warning" src="https://raw.githubusercontent.com/Mqxx/GitHub-Markdown/main/blockquotes/badge/dark-theme/warning.svg">
> </picture><br>
>
> `np.resize()` and `ndarray.resize()` are different, `np.resize()` returns a new array with the new shape and `arr.resize()` changes the shape of the array in place, also `np.resize()` can take a tuple as the new shape and fills the remaining elements with *copies* while `arr.resize()` can only take a list as the new shape and fills the remaining elements with *zeros*.

### Broadcasting

Broadcasting is the mechanism that allows NumPy to perform array arithmetic between `ndarrays` of different size or shape. The way broadcasting works is by first comparing the shape of the two arrays element-wise starting from the trailing dimensions and working its way forward. Two dimensions are compatible when they are equal or one of them is 1. If these conditions are not met then a `ValueError` is raised. After the dimensions are found to be compatible, the array with fewer dimensions is padded with ones on its leading side. After that, the arrays are said to be `broadcastable` and the broadcasting operation is performed on them. 

For example, we can add a scalar to an array of any shape, or we can add a 1D array to a 2D array. 

[Example of broadcasting (Vector Quantization Algorithm))](https://numpy.org/doc/stable/user/basics.broadcasting.html#a-practical-example-vector-quantization)

### Universal Functions (UFuncs) and Aggregation Functions

Numpy is heavily optimized for performing `Vectorized` operations. In numpy it is implemented using `UFuncs` (Universal functions). Some `UFuncs` are;

- `np.add(x1, x2)` : Add two arrays element-wise.

- `np.subtract(x1, x2)` : Subtract elements of second array from first array.

- `np.multiply(x1, x2)` : Multiply two arrays element-wise.

- `np.divide(x1, x2)` : Divide first array by second array element-wise.

- `np.power(x1, x2)` : Raise elements of first array to powers from second array element-wise.

- `np.log(x)` : Natural logarithm, element-wise.

- `np.sin(x)` : Computes the sine of each element in radians.

- `np.cos(x)` : Computes the cosine of each element in radians.

- `np.round(x)` : Rounds each element to the nearest integer.

`Aggregate/reduction Functions` are the functions in numpy that can be used to perform operations on the entire array. Some of the aggregate functions are:

- `np.sum()`
- `np.min()`
- `np.max()`
- `np.mean()`
- `np.any()`
- `np.all()`

> All them supports operations along a specified axis. The default axis is 0, which is the row axis.  

![Axis representation of numpy](image-1.png)

<!-- Reviewed -->

### Views and Copies

Numpy saves memory whenever possible using by directly using a `View` instead of passing copies. It does this by accessing the internal data buffer. A `View` can be created using `np.view()` or by slicing an array. Although this enures good performance, it can become a problem if we use numpy arrays without knowing this.

We can find whether an array is a view or a copy by using `np.base` which returns the base object of the array. If the array is a view then it will return the original array, if it is a copy then it will return `None`.

```python
arr  = np.array([1, 2, 3, 4, 5])
arr2 = arr[1:4]
print(arr2.base) # [1 2 3 4 5]
```

### Structured arrays

Structured arrays are nd-arrays whose datatype is a composition of simpler data types organized as a sequence of named fields. They are designed to mimic `structs` in C. 

```python
dt = np.dtype([('name', np.unicode_, 16), ('grades', np.float64, (2,))])
x = np.array([('Sarah', (8.0, 7.0)), ('John', (6.0, 7.0))], dtype=dt)
print(x[1]) # ('John', [6., 7.])
print(x['grades']) # [array([8., 7.]) array([6., 7.])]
```

### [Numpy Documentation](https://numpy.org/doc/stable/) | [Numpy tutorials](https://numpy.org/numpy-tutorials/) | [Numpy Case study](./Case_study.ipynb) 

**[Numpy Exercises](./Exercises.ipynb) | [Workbook 1](./Numpy-I.ipynb) | [Workbook 2](./Numpy-II.ipynb) | [Workbook 3](./Numpy-III.ipynb)**
