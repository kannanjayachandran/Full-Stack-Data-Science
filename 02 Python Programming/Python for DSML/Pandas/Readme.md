<!-- 
    Author : Kannan Jayachandran
    File : Readme.md (Pandas)
    Section : Python for DSML
 -->

<h1 align="center"> Pandas </h1>

## Table of Contents

1. [Introduction](#introduction)
2. [Core Data Structures](#core-data-structures)
3. [Series: One-Dimensional Data](#series-one-dimensional-data)
4. [DataFrame: Two-Dimensional Data](#dataframe-two-dimensional-data)
5. [Axes in Pandas](#axes-in-pandas)
6. [Indexing and Selection](#indexing-and-selection)
7. [Data Types and Memory Management](#data-types-and-memory-management)
8. [Data Input/Output](#data-inputoutput)
9. [Data Inspection and Exploration](#data-inspection-and-exploration)
10. [Handling Missing Data](#handling-missing-data)
11. [Data Manipulation and Transformation](#data-manipulation-and-transformation)
12. [Grouping and Aggregation](#grouping-and-aggregation)
13. [Merging, Joining, and Concatenation](#merging-joining-and-concatenation)
14. [Time Series Operations](#time-series-operations)
15. [String and Categorical Data](#string-and-categorical-data)
16. [Performance Optimization](#performance-optimization)
17. [Best Practices and Common Pitfalls](#best-practices-and-common-pitfalls)

---

## Introduction

**Pandas** is the foundational library for **data manipulation and analysis** in Python, built on top of NumPy. Originally developed for financial modeling, it excels at handling structured, tabular data with powerful tools for cleaning, transforming, and analyzing datasets.

### Why Pandas for Data Science?

1.  **Intuitive Data Structures**: Series (1D) and DataFrame (2D) mirror real-world data
2. **SQL-like Operations**: Grouping, joining, merging without a database
3. **Missing Data Handling**: Robust tools for NA/NaN values
4. **Time Series**: Native datetime support with resampling, windowing, and frequency conversion
5. **Heterogeneous Data**: Different data types per column (unlike NumPy)
6. **Integration**: Seamless with NumPy, Matplotlib, Scikit-learn, SQL databases
7. **Versatile I/O**: Read/write CSV, Excel, SQL, JSON, Parquet, HDF5, etc.

### Installation

```bash
pip install pandas
```

---

## Core Data Structures

Pandas provides two primary data structures that form the foundation of all data operations:

- **Series**
- **DataFrame**

![How series are DataFrame are related to other data storage sources](./img/series_df__compare.png)

**Comparison with Other Data Structures**

| Structure | Dimensions | Homogeneous | Mutable Size | Index Labels |
|-----------|-----------|-------------|--------------|--------------|
| NumPy Array | N-D | Yes | No | Integer only |
| Pandas Series | 1-D | Yes | No | Any hashable |
| Pandas DataFrame | 2-D | No (per column) | Yes | Any hashable |
| Python List | 1-D | No | Yes | Integer only |
| Python Dict | 1-D | No | Yes | Any hashable |

![series and dataFrame in pandas](./img/pd_data_structures.png)

---

## Series: One-Dimensional Data

A **Series** is a one-dimensional labeled array capable of holding any data type.

It can be thought of as a **fixed-size ordered dictionary** mapping labels (index) to values.

### Creating Series

```python
import pandas as pd
import numpy as np

# From list (default integer index)
s1 = pd.Series([10, 20, 30, 40])
print(s1)
# 0    10
# 1    20
# 2    30
# 3    40
# dtype: int64

# With custom index
s2 = pd.Series([10, 20, 30, 40], 
               index=['a', 'b', 'c', 'd'],
               name='values')

# From dictionary (keys become index)
data_dict = {'US': 46, 'China': 38, 'Japan': 27, 'GB': 22}
medals = pd.Series(data_dict, name='Gold_Medals')

# From scalar (broadcasts to all indices)
s3 = pd.Series(5.0, index=['a', 'b', 'c', 'd'])

# From NumPy array
arr = np.random.randn(5)
s4 = pd.Series(arr, index=['v', 'w', 'x', 'y', 'z'])
```

### Series Attributes

```python
medals = pd.Series({'US': 46, 'China': 38, 'Japan': 27, 'GB': 22}, 
                   name='Gold_Medals')

# Core attributes
print(f"Values: {medals.values}")        # NumPy array
print(f"Index: {medals.index}")          # Index object
print(f"Name: {medals.name}")            # 'Gold_Medals'
print(f"Dtype: {medals.dtype}")          # int64
print(f"Size: {medals.size}")            # 4 (total elements including NaN)
print(f"Count: {medals.count()}")        # 4 (non-NaN values)
print(f"Shape: {medals.shape}")          # (4,)
print(f"Memory: {medals.memory_usage()} bytes")
```

### Series Operations

```python
# Arithmetic operations (element-wise)
s = pd.Series([1, 2, 3, 4])
print(s + 10)      # Add scalar
print(s * 2)       # Multiply
print(s ** 2)      # Power

# Operations between Series (aligns on index)
s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
result = s1 + s2
# a     11
# b     22
# c     33
# d    NaN  (no matching index in s1)

# Aggregations
data = pd.Series([1, 2, 3, 4, 5])
print(f"Sum: {data.sum()}")
print(f"Mean: {data.mean()}")
print(f"Std: {data.std()}")
print(f"Min: {data.min()}")
print(f"Max: {data.max()}")
print(f"Median: {data.median()}")

# Statistical summary
print(data.describe())
```

### Series Indexing

```python
s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])

# By label (explicit indexing)
print(s['b'])           # 20
print(s[['a', 'c']])    # Multiple labels

# By position (implicit indexing)
print(s.iloc[1])        # 20
print(s.iloc[0:2])      # First two elements

# Boolean indexing
print(s[s > 20])        # Values > 20

# Fancy indexing
indices = [0, 2]
print(s.iloc[indices])
```

**Important**: When index is integer, `s[2]` uses explicit (label) indexing, not positional!

### Series Methods

```python
s = pd.Series([3, 1, 4, 1, 5, 9, 2, 6], name='numbers')

# Sorting
print(s.sort_values())                    # Sort by values
print(s.sort_index())                     # Sort by index

# Unique values
print(s.unique())                         # Array of unique values
print(s.nunique())                        # Count of unique values
print(s.value_counts())                   # Frequency of each value

# Ranking
print(s.rank())                           # Assign ranks to values

# Checking for duplicates
print(s.duplicated())                     # Boolean mask
print(s.drop_duplicates())                # Remove duplicates

# String operations (if dtype is object/string)
s_str = pd.Series(['apple', 'banana', 'cherry'])
print(s_str.str.upper())                  # 'APPLE', 'BANANA', 'CHERRY'
print(s_str.str.contains('an'))           # Boolean mask

# Apply custom function
def square_if_even(x):
    return x**2 if x % 2 == 0 else x

print(s.apply(square_if_even))

# Map values (dictionary-based transformation)
mapping = {1: 'one', 2: 'two', 3: 'three'}
print(s.map(mapping))                     # NaN for unmapped values
```

* Pandas Series supports vectorized operations like `Numpy` arrays.

---

## DataFrame: Two-Dimensional Data

A **DataFrame** is a **2D labeled data structure** with rows and columns, similar to SQL tables or spreadsheets. Each column is a Series, potentially of a different data type.

### Conceptual Structure

```python
# Conceptual representation of DataFrame internals
df_concept = {
    'index': [0, 1, 2],
    'columns': [
        {'name': 'growth', 'data': pd.Series([0.5, 0.7, 1.2])},
        {'name': 'revenue', 'data': pd.Series([100, 150, 200])}
    ]
}
```

### Creating DataFrames

```python
# From dictionary of lists (most common)
data = {
    'Country': ['US', 'China', 'Japan', 'GB'],
    'Gold': [46, 38, 27, 22],
    'Silver': [37, 32, 14, 21],
    'Bronze': [38, 18, 17, 22]
}
df = pd.DataFrame(data)

# From list of dictionaries
records = [
    {'Country': 'US', 'Gold': 46, 'Silver': 37},
    {'Country': 'China', 'Gold': 38, 'Silver': 32},
    {'Country': 'Japan', 'Gold': 27, 'Silver': 14}
]
df = pd.DataFrame(records)

# From NumPy array with column names
arr = np.random.randn(4, 3)
df = pd.DataFrame(arr, 
                  columns=['A', 'B', 'C'],
                  index=['w', 'x', 'y', 'z'])

# From dictionary of Series
s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
df = pd.DataFrame({'col1': s1, 'col2': s2})

# From nested lists
data = [
    ['US', 46, 37, 38],
    ['China', 38, 32, 18],
    ['Japan', 27, 14, 17]
]
df = pd.DataFrame(data, columns=['Country', 'Gold', 'Silver', 'Bronze'])

# With custom index
df = pd.DataFrame(data, 
                  columns=['Country', 'Gold', 'Silver', 'Bronze'],
                  index=['first', 'second', 'third'])
```

### DataFrame Attributes

```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4.0, 5.0, 6.0],
    'C': ['x', 'y', 'z']
})

# Core attributes
print(f"Shape: {df.shape}")              # (3, 3) - (rows, cols)
print(f"Size: {df.size}")                # 9 - total elements
print(f"Columns: {df.columns}")          # Column labels
print(f"Index: {df.index}")              # Row labels
print(f"Dtypes:\n{df.dtypes}")           # Data type per column
print(f"Values:\n{df.values}")           # NumPy array (2D)
print(f"Memory usage:\n{df.memory_usage()}")
print(f"Memory (deep): {df.memory_usage(deep=True).sum()} bytes")

# Axes (dimension names)
print(f"Axes: {df.axes}")                # [index, columns]
```

### DataFrame Methods Overview

Pandas DataFrame methods can be categorized into several types:

#### 1. Dunder Methods (Special Methods)
```python
# Arithmetic operations (return DataFrame)
df1 + df2          # __add__
df1 - df2          # __sub__
df1 * df2          # __mul__
df1 / df2          # __truediv__

# Iteration
for col in df:     # __iter__ (iterates over column names)
    print(col)

# Indexing
df['A']            # __getitem__
df['A'] = [1,2,3]  # __setitem__
```

#### 2. Aggregate Methods (Reduce to Scalar/Series)
```python
df.mean()          # Mean of each column
df.sum()           # Sum of each column
df.max()           # Maximum value per column
df.min()           # Minimum value per column
df.std()           # Standard deviation per column
df.var()           # Variance per column
df.median()        # Median per column
df.count()         # Non-null count per column

# Aggregate multiple functions
df.agg(['mean', 'std', 'min', 'max'])

# Per-column aggregation
df.agg({'A': 'sum', 'B': 'mean', 'C': 'count'})
```

#### 3. Conversion Methods (Export Data)
```python
df.to_csv('data.csv')          # Export to CSV
df.to_excel('data.xlsx')       # Export to Excel
df.to_json('data.json')        # Export to JSON
df.to_sql('table', con)        # Export to SQL database
df.to_parquet('data.parquet')  # Export to Parquet
df.to_dict()                   # Convert to dictionary
df.to_numpy()                  # Convert to NumPy array
df.to_html()                   # Convert to HTML table
df.to_markdown()               # Convert to Markdown table
```

#### 4. Manipulation Methods (Return Modified DataFrame)
```python
df.sort_values(by='A')         # Sort by column values
df.sort_index()                # Sort by index
df.drop_duplicates()           # Remove duplicate rows
df.drop(columns=['A'])         # Drop columns
df.dropna()                    # Drop rows with NaN
df.fillna(0)                   # Fill NaN values
df.replace(old, new)           # Replace values
df.rename(columns={'A': 'X'})  # Rename columns
df.set_index('A')              # Set column as index
df.reset_index()               # Reset index to default
```

#### 5. Indexing and Accessor Methods
```python
df.loc[row_label, col_label]   # Label-based indexing
df.iloc[row_pos, col_pos]      # Position-based indexing
df.at[row, col]                # Fast scalar access (label)
df.iat[row, col]               # Fast scalar access (position)
df.query('A > 5')              # Query using expression
```

#### 6. Accessor Methods (`.str`, `.dt`, `.cat`)
```python
# String operations
df['Name'].str.upper()         # Uppercase all strings
df['Name'].str.contains('John')  # Boolean mask

# Datetime operations
df['Date'].dt.year             # Extract year
df['Date'].dt.month            # Extract month
df['Date'].dt.dayofweek        # Day of week

# Categorical operations
df['Category'].cat.codes       # Numeric codes
df['Category'].cat.categories  # Category labels
```

#### 7. Transformation Methods
```python
df.apply(func)                 # Apply function to each column
df.applymap(func)              # Apply function to each element (deprecated, use map)
df.map(func)                   # Apply function to each element
df.transform(func)             # Transform each group
df.agg(func)                   # Aggregate using function
df.pivot()                     # Reshape data
df.pivot_table()               # Create pivot table
df.melt()                      # Unpivot DataFrame
df.stack()                     # Pivot columns to rows
df.unstack()                   # Pivot rows to columns
```

#### 8. Grouping Methods
```python
df.groupby('Category')         # Group by column
df.groupby(['A', 'B'])         # Group by multiple columns
df.groupby('A').sum()          # Aggregate after grouping
df.groupby('A').agg({'B': 'sum', 'C': 'mean'})
```

---

## Axes in Pandas

Understanding axes is crucial for operations on DataFrames and Series. Unlike NumPy (where axis 0 is down rows), Pandas uses:

- **Axis 0 (Index)**: Represents **rows** (down the DataFrame)
- **Axis 1 (Columns)**: Represents **columns** (across the DataFrame)

**Mnemonic**: 
- `axis=0`: "Down the index" → operates on rows, returns column-wise results
- `axis=1`: "Across the columns" → operates on columns, returns row-wise results

![Axes](./img/Join_data_along_axis.png)

### Axis Operations Examples

```python
data = {
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8],
    'C': [9, 10, 11, 12]
}
df = pd.DataFrame(data)

# Axis 0: Column-wise aggregation (operate down rows)
print(df.sum(axis=0))
# A     10
# B     26
# C     42

# Axis 1: Row-wise aggregation (operate across columns)
print(df.sum(axis=1))
# 0    15
# 1    18
# 2    21
# 3    24

# Drop operation
df.drop('A', axis=1)       # Drop column A
df.drop(0, axis=0)         # Drop row at index 0

# Mean calculation
df.mean(axis=0)            # Mean of each column
df.mean(axis=1)            # Mean of each row

# Concatenation
pd.concat([df1, df2], axis=0)  # Stack vertically (add rows)
pd.concat([df1, df2], axis=1)  # Stack horizontally (add columns)
```

### Common Confusion: Index vs Axis

```python
df = pd.DataFrame({
    'X': [1, 2, 3],
    'Y': [4, 5, 6]
}, index=['a', 'b', 'c'])

# These are equivalent:
df.drop('a', axis=0)       # Drop row 'a'
df.drop(index='a')         # More explicit

df.drop('X', axis=1)       # Drop column 'X'
df.drop(columns='X')       # More explicit

# Sum along axis
df.sum(axis=0)             # Sum each column (result: Series with column names)
df.sum(axis=1)             # Sum each row (result: Series with index labels)
```

**Best Practice**: Use explicit `index=` or `columns=` parameters instead of `axis` for clarity.

---

## Indexing and Selection

Pandas provides multiple ways to access data. Understanding the difference between **implicit** (positional) and **explicit** (label-based) indexing is critical.

### Index Types

```python
# Default integer index (RangeIndex)
df1 = pd.DataFrame({'A': [1, 2, 3]})
print(df1.index)  # RangeIndex(start=0, stop=3, step=1)

# Custom string index
df2 = pd.DataFrame({'A': [1, 2, 3]}, index=['x', 'y', 'z'])
print(df2.index)  # Index(['x', 'y', 'z'], dtype='object')

# Integer index (not RangeIndex)
df3 = pd.DataFrame({'A': [1, 2, 3]}, index=[10, 20, 30])
print(df3.index)  # Int64Index([10, 20, 30])

# DatetimeIndex
dates = pd.date_range('2024-01-01', periods=3)
df4 = pd.DataFrame({'A': [1, 2, 3]}, index=dates)
print(df4.index)  # DatetimeIndex(...)

# MultiIndex (hierarchical)
arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
index = pd.MultiIndex.from_arrays(arrays, names=['letter', 'number'])
df5 = pd.DataFrame({'value': [10, 20, 30, 40]}, index=index)
```

### Indexing Principles

**Key Rule**: 
- **Single label** → selects column (returns Series)
- **List of labels** → selects columns (returns DataFrame)
- **Slice** → selects rows (returns DataFrame)
- **Boolean array** → selects rows (returns DataFrame)

```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
}, index=['x', 'y', 'z'])

# Column selection
df['A']              # Series (single column)
df[['A', 'C']]       # DataFrame (multiple columns)

# Row selection (slicing)
df[0:2]              # First 2 rows (implicit indexing)
df['x':'y']          # Rows 'x' to 'y' INCLUSIVE (explicit indexing)

# Boolean indexing (rows)
df[df['A'] > 1]      # Rows where A > 1
```

### `.loc` - Label-Based Indexing

`.loc` uses **explicit** index labels. **Slicing is inclusive on both ends**.

```python
df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8]
}, index=['w', 'x', 'y', 'z'])

# Single row
df.loc['x']              # Series (row 'x')

# Multiple rows
df.loc[['w', 'z']]       # DataFrame (rows 'w' and 'z')

# Slice (INCLUSIVE)
df.loc['w':'y']          # Rows 'w', 'x', 'y' (inclusive!)

# Row and column
df.loc['x', 'A']         # Scalar (row 'x', column 'A')

# Rows and columns
df.loc[['w', 'y'], ['A']]  # DataFrame subset

# All rows, specific columns
df.loc[:, 'A']           # Series (all rows, column 'A')
df.loc[:, ['A', 'B']]    # DataFrame

# Boolean indexing with loc
df.loc[df['A'] > 2]      # Rows where A > 2

# Setting values
df.loc['x', 'A'] = 100   # Set single value
df.loc[df['A'] > 2, 'B'] = 0  # Conditional assignment
```

### `.iloc` - Position-Based Indexing

`.iloc` uses **implicit** integer positions (0-indexed). **Slicing excludes the end**.

```python
df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8]
}, index=['w', 'x', 'y', 'z'])

# Single row (by position)
df.iloc[1]               # Series (second row)

# Multiple rows
df.iloc[[0, 2]]          # DataFrame (1st and 3rd rows)

# Slice (EXCLUSIVE)
df.iloc[0:2]             # Rows 0, 1 (excludes 2)

# Row and column by position
df.iloc[1, 0]            # Scalar (2nd row, 1st column)

# Rows and columns
df.iloc[[0, 2], [1]]     # DataFrame subset

# Negative indexing
df.iloc[-1]              # Last row
df.iloc[:, -1]           # Last column

# Boolean array (by position)
mask = [True, False, True, False]
df.iloc[mask]            # Rows 0 and 2
```

### `.at` and `.iat` - Fast Scalar Access

For **single value** access/assignment, `.at` and `.iat` are faster than `.loc` and `.iloc`.

```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
}, index=['x', 'y', 'z'])

# Fast label-based scalar access
value = df.at['y', 'A']        # 2 (faster than df.loc['y', 'A'])
df.at['y', 'A'] = 100          # Fast assignment

# Fast position-based scalar access
value = df.iat[1, 0]           # 2 (faster than df.iloc[1, 0])
df.iat[1, 0] = 100             # Fast assignment
```

**Use Case**: When accessing/setting single values in loops (though vectorized operations are preferred).

### Boolean Indexing

```python
df = pd.DataFrame({
    'Age': [25, 30, 35, 40],
    'Salary': [50000, 60000, 70000, 80000],
    'City': ['NY', 'LA', 'NY', 'SF']
}, index=['Alice', 'Bob', 'Charlie', 'David'])

# Single condition
high_salary = df[df['Salary'] > 60000]

# Multiple conditions (use & | ~, not and or not)
young_high_earners = df[(df['Age'] < 35) & (df['Salary'] > 55000)]

# isin() for membership
ny_or_sf = df[df['City'].isin(['NY', 'SF'])]

# String methods
ny_residents = df[df['City'].str.contains('NY')]

# between() for range checking
mid_age = df[df['Age'].between(28, 38)]

# Negation
not_ny = df[~df['City'].str.contains('NY')]

# Query method (SQL-like syntax)
result = df.query('Age > 30 and Salary < 75000')
result = df.query('City == "NY"')
```

**Important**: Always use `&` (and), `|` (or), `~` (not) with parentheses around each condition!

### Indexing with Duplicate Labels

```python
# Pandas supports duplicate index values
df = pd.DataFrame({
    'A': [1, 2, 3, 4]
}, index=['x', 'y', 'x', 'z'])

# Returns all rows with label 'x' (DataFrame, not Series!)
print(df.loc['x'])
#    A
# x  1
# x  3

# Check for duplicates
print(df.index.is_unique)  # False
print(df.index.duplicated())  # [False, False, True, False]
```

**Best Practice**: Avoid duplicate indices unless there's a specific reason (e.g., time series with repeated timestamps).

---

## Data Types and Memory Management

### Pandas Data Types

Pandas extends NumPy's type system with additional types for better data representation:

| Pandas Type | NumPy Equivalent | Description | Example |
|-------------|------------------|-------------|---------|
| `int64`, `int32`, `int16`, `int8` | `int64`, etc. | Signed integers | Age, count data |
| `uint64`, `uint32`, `uint16`, `uint8` | `uint64`, etc. | Unsigned integers | IDs, positive counts |
| `float64`, `float32`, `float16` | `float64`, etc. | Floating point | Prices, measurements |
| `bool` | `bool` | Boolean | Flags, conditions |
| `object` | `object` | Python objects | Mixed types, strings |
| `string` | - | Dedicated string type | Text data (preferred over object) |
| `category` | - | Categorical data | Fixed set of values |
| `datetime64[ns]` | `datetime64` | Datetimes | Timestamps |
| `timedelta64[ns]` | `timedelta64` | Time durations | Time differences |
| `Int64`, `Int32`, etc. | - | **Nullable integers** | Integers with NaN support |
| `Float64`, `Float32` | - | Nullable floats | Explicit nullable floats |
| `boolean` | - | Nullable booleans | Booleans with NaN support |

### Checking and Converting Data Types

```python
df = pd.DataFrame({
    'int_col': [1, 2, 3],
    'float_col': [1.1, 2.2, 3.3],
    'str_col': ['a', 'b', 'c'],
    'mixed_col': [1, 'a', 3.5]
})

# Check dtypes
print(df.dtypes)
print(df['int_col'].dtype)

# Convert types
df['int_col'] = df['int_col'].astype('float64')
df['str_col'] = df['str_col'].astype('string')  # Preferred over 'object'

# Multiple conversions
df = df.astype({
    'int_col': 'int32',
    'float_col': 'float32'
})

# Automatic type inference
df = df.infer_objects()

# Convert to best possible dtypes
df = df.convert_dtypes()  # Uses nullable types when appropriate
```

### Category Data Type

**Use Case**: Columns with a limited set of unique values (gender, country, product category).

**Benefits**:
- Reduced memory usage (stores as integers internally)
- Faster operations (groupby, value_counts)
- Enforces valid values
- Preserves ordering (if needed)

```python
# Regular object dtype
df = pd.DataFrame({
    'city': ['NYC', 'LA', 'NYC', 'SF', 'LA', 'NYC'] * 10000
})
print(f"Object dtype: {df.memory_usage(deep=True)['city'] / 1e6:.2f} MB")

# Convert to category
df['city'] = df['city'].astype('category')
print(f"Category dtype: {df.memory_usage(deep=True)['city'] / 1e6:.2f} MB")

# Create ordered category
df['size'] = pd.Categorical(['S', 'M', 'L', 'M', 'S'], 
                             categories=['S', 'M', 'L', 'XL'],
                             ordered=True)

# Category operations
print(df['city'].cat.categories)      # Unique categories
print(df['city'].cat.codes)           # Integer codes
df['city'] = df['city'].cat.add_categories(['Boston'])
df['city'] = df['city'].cat.remove_unused_categories()

# Ordered category comparisons
print(df[df['size'] >= 'M'])  # Works with ordered categories
```

**Memory Savings Example**:
```python
# 1 million rows with 5 unique values
import numpy as np

# Object dtype
df_obj = pd.DataFrame({
    'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], 1_000_000)
})
print(f"Object: {df_obj.memory_usage(deep=True)['category'] / 1e6:.2f} MB")

# Category dtype
df_cat = pd.DataFrame({
    'category': pd.Categorical(np.random.choice(['A', 'B', 'C', 'D', 'E'], 1_000_000))
})
print(f"Category: {df_cat.memory_usage(deep=True)['category'] / 1e6:.2f} MB")
print(f"Memory reduction: {(1 - df_cat.memory_usage(deep=True)['category'] / df_obj.memory_usage(deep=True)['category']) * 100:.1f}%")
```

### Nullable Integer Data Type

**Problem**: NumPy arrays (and standard Pandas dtypes) cannot have NaN in integer arrays. Missing values force conversion to float64.

```python
# Traditional problem
data = pd.Series([1, 2, None, 4])
print(data.dtype)  # float64 (forced conversion due to NaN)

# Solution: Nullable integer types
data_nullable = pd.Series([1, 2, None, 4], dtype='Int64')  # Capital 'I'
print(data_nullable.dtype)  # Int64 (maintains integer type)
print(data_nullable)
# 0       1
# 1       2
# 2    <NA>
# 3       4

# Works with operations
print(data_nullable + 10)
# 0      11
# 1      12
# 2    <NA>
# 3      14

# Useful for ID columns
df = pd.DataFrame({
    'user_id': pd.array([101, 102, None, 104], dtype='Int64'),
    'age': pd.array([25, None, 35, 40], dtype='Int64')
})
```

**Available Nullable Types**: `Int8`, `Int16`, `Int32`, `Int64`, `UInt8`, `UInt16`, `UInt32`, `UInt64`, `Float32`, `Float64`, `boolean`, `string`

### Memory Optimization Strategies

```python
# Check memory usage
print(df.info(memory_usage='deep'))
print(df.memory_usage(deep=True))

# Optimize dtypes automatically
def optimize_dtypes(df):
    """Reduce memory by downcasting numeric types"""
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df

# Example: Optimize large dataset
df_large = pd.DataFrame({
    'id': range(1_000_000),
    'age': np.random.randint(0, 100, 1_000_000),
    'score': np.random.randn(1_000_000)
})

print(f"Before: {df_large.memory_usage(deep=True).sum() / 1e6:.2f} MB")
df_large = optimize_dtypes(df_large)
print(f"After: {df_large.memory_usage(deep=True).sum() / 1e6:.2f} MB")

# Read CSV with optimized dtypes
df = pd.read_csv('data.csv', dtype={
    'id': 'int32',
    'category': 'category',
    'amount': 'float32'
})

# Use chunking for very large files
chunk_iter = pd.read_csv('huge_file.csv', chunksize=10000)
for chunk in chunk_iter:
    process_chunk(chunk)  # Process in batches
```

---

## Data Input/Output

Pandas excels at reading and writing data from various formats. Understanding these operations is crucial for real-world data science workflows.

### Reading CSV Files

```python
# Basic CSV reading
df = pd.read_csv('data.csv')

# Common parameters for CSV
df = pd.read_csv(
    'data.csv',
    sep=',',                    # Delimiter (also: '\t', '|', etc.)
    header=0,                   # Row number for column names (None if no header)
    names=['col1', 'col2'],     # Custom column names
    index_col='id',             # Column to use as index
    usecols=['col1', 'col2'],   # Only read specific columns
    dtype={'col1': 'int32'},    # Specify dtypes
    parse_dates=['date_col'],   # Parse as datetime
    na_values=['NA', 'missing'], # Additional NA values
    skiprows=5,                 # Skip first 5 rows
    nrows=1000,                 # Read only first 1000 rows
    encoding='utf-8',           # Character encoding
    low_memory=False            # Read entire file into memory (for dtype inference)
)

# Handle different delimiters
df = pd.read_csv('data.tsv', sep='\t')  # Tab-separated
df = pd.read_csv('data.txt', sep='|')   # Pipe-separated

# Read from URL
df = pd.read_csv('https://example.com/data.csv')

# Read compressed files
df = pd.read_csv('data.csv.gz', compression='gzip')
df = pd.read_csv('data.csv.zip', compression='zip')

# Read with specific date parsing
df = pd.read_csv('data.csv', 
                 parse_dates=['date'],
                 date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d'))

# Memory-efficient reading (chunking)
chunk_size = 10000
chunks = []
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process each chunk
    chunks.append(chunk[chunk['value'] > 0])
df = pd.concat(chunks, ignore_index=True)
```

### Writing CSV Files

```python
# Basic CSV writing
df.to_csv('output.csv')

# Common parameters
df.to_csv(
    'output.csv',
    sep=',',                    # Delimiter
    index=False,                # Don't write row indices
    columns=['col1', 'col2'],   # Write specific columns only
    header=True,                # Include column names
    na_rep='NULL',              # String representation of NaN
    encoding='utf-8',           # Character encoding
    mode='a',                   # Append mode ('w' for overwrite)
    compression='gzip'          # Compress output
)

# Write without index (most common)
df.to_csv('output.csv', index=False)

# Append to existing file
df.to_csv('output.csv', mode='a', header=False, index=False)

# Write to buffer (for uploading, etc.)
from io import StringIO
buffer = StringIO()
df.to_csv(buffer, index=False)
csv_string = buffer.getvalue()
```

### Reading Excel Files

```python
# Basic Excel reading
df = pd.read_excel('data.xlsx')

# Specify sheet
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df = pd.read_excel('data.xlsx', sheet_name=0)  # By index

# Read multiple sheets
dfs = pd.read_excel('data.xlsx', sheet_name=None)  # Returns dict
df1 = dfs['Sheet1']
df2 = dfs['Sheet2']

# Common parameters
df = pd.read_excel(
    'data.xlsx',
    sheet_name='Data',
    header=0,
    names=['col1', 'col2'],
    index_col=0,
    usecols='A:D',              # Excel-style columns
    dtype={'col1': 'str'},
    parse_dates=['date_col'],
    skiprows=5,
    nrows=100
)

# Read specific cell range
df = pd.read_excel('data.xlsx', usecols='B:E', skiprows=3, nrows=20)

# Read .xls (older format)
df = pd.read_excel('data.xls', engine='xlrd')
```

### Writing Excel Files

```python
# Basic Excel writing
df.to_excel('output.xlsx', index=False)

# Write multiple sheets
with pd.ExcelWriter('output.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Data1', index=False)
    df2.to_excel(writer, sheet_name='Data2', index=False)

# Advanced Excel writing with formatting
with pd.ExcelWriter('output.xlsx', engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name='Data', index=False)
    
    # Access workbook and worksheet
    workbook = writer.book
    worksheet = writer.sheets['Data']
    
    # Add formatting
    header_format = workbook.add_format({'bold': True, 'bg_color': '#D3D3D3'})
    worksheet.set_row(0, None, header_format)
    worksheet.set_column('A:A', 15)  # Set column width

# Append to existing Excel file
with pd.ExcelWriter('output.xlsx', mode='a', engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='NewSheet', index=False)
```

### Reading SQL Databases

```python
import sqlite3
from sqlalchemy import create_engine

# Using sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql('SELECT * FROM table_name', conn)
df = pd.read_sql('SELECT * FROM table WHERE age > 25', conn)
conn.close()

# Using SQLAlchemy (supports multiple databases)
engine = create_engine('postgresql://user:password@localhost:5432/dbname')
df = pd.read_sql('SELECT * FROM users', engine)

# Read entire table
df = pd.read_sql_table('table_name', engine)

# Read with query
query = """
    SELECT u.name, o.amount 
    FROM users u 
    JOIN orders o ON u.id = o.user_id 
    WHERE o.amount > 100
"""
df = pd.read_sql_query(query, engine)

# Chunking for large queries
for chunk in pd.read_sql('SELECT * FROM large_table', engine, chunksize=10000):
    process(chunk)

# With index
df = pd.read_sql('SELECT * FROM users', engine, index_col='user_id')

# Parse dates
df = pd.read_sql('SELECT * FROM events', engine, parse_dates=['event_date'])
```

### Writing to SQL Databases

```python
import sqlite3
from sqlalchemy import create_engine

# Using sqlite3
conn = sqlite3.connect('database.db')
df.to_sql('table_name', conn, if_exists='replace', index=False)
conn.close()

# Using SQLAlchemy
engine = create_engine('postgresql://user:password@localhost:5432/dbname')

# Write to table
df.to_sql(
    'users',
    engine,
    if_exists='replace',    # 'fail', 'replace', or 'append'
    index=False,
    dtype={                 # Specify SQL types
        'id': 'INTEGER',
        'name': 'VARCHAR(100)',
        'created_at': 'TIMESTAMP'
    },
    method='multi',         # Faster bulk insert
    chunksize=1000         # Insert in batches
)

# Append data
df.to_sql('users', engine, if_exists='append', index=False)

# Write with index
df.to_sql('users', engine, index=True, index_label='user_id')
```

### Reading JSON Files

```python
# Basic JSON reading
df = pd.read_json('data.json')

# Different orientations
df = pd.read_json('data.json', orient='records')
# [{"a": 1, "b": 2}, {"a": 3, "b": 4}]

df = pd.read_json('data.json', orient='columns')
# {"a": [1, 3], "b": [2, 4]}

df = pd.read_json('data.json', orient='index')
# {"row1": {"a": 1, "b": 2}, "row2": {"a": 3, "b": 4}}

df = pd.read_json('data.json', orient='split')
# {"columns": ["a", "b"], "index": [0, 1], "data": [[1, 2], [3, 4]]}

df = pd.read_json('data.json', orient='values')
# [[1, 2], [3, 4]]

# Read JSON lines (newline-delimited JSON)
df = pd.read_json('data.jsonl', lines=True)

# Read from string
json_string = '{"a": [1, 2, 3], "b": [4, 5, 6]}'
df = pd.read_json(json_string)

# From URL
df = pd.read_json('https://api.example.com/data.json')

# Parse dates
df = pd.read_json('data.json', convert_dates=['date_column'])
```

### Writing JSON Files

```python
# Basic JSON writing
df.to_json('output.json')

# Different orientations
df.to_json('output.json', orient='records')  # List of dicts (most common)
df.to_json('output.json', orient='columns')  # Dict of lists
df.to_json('output.json', orient='index')    # Dict of dicts
df.to_json('output.json', orient='split')    # Dict with separate keys
df.to_json('output.json', orient='values')   # Just the values

# Pretty printing
df.to_json('output.json', indent=4)

# JSON lines format
df.to_json('output.jsonl', orient='records', lines=True)

# Control date formatting
df.to_json('output.json', date_format='iso')  # ISO 8601
df.to_json('output.json', date_format='epoch')  # Unix timestamp
```

### Other File Formats

```python
# Parquet (columnar format, very efficient)
df.to_parquet('data.parquet', compression='snappy')
df = pd.read_parquet('data.parquet')

# HDF5 (hierarchical data format)
df.to_hdf('data.h5', key='data', mode='w')
df = pd.read_hdf('data.h5', key='data')

# Feather (fast, language-agnostic)
df.to_feather('data.feather')
df = pd.read_feather('data.feather')

# Pickle (Python-specific, preserves all dtypes)
df.to_pickle('data.pkl')
df = pd.read_pickle('data.pkl')

# HTML tables
df.to_html('table.html', index=False)
dfs = pd.read_html('https://example.com/tables.html')  # Returns list

# Clipboard (useful for quick copy-paste)
df.to_clipboard(index=False)
df = pd.read_clipboard()

# Markdown
print(df.to_markdown())  # For documentation
```

### Best Practices for I/O

```python
# 1. Always specify dtypes when reading large files
df = pd.read_csv('large_file.csv', dtype={'id': 'int32', 'category': 'category'})

# 2. Use categorical for repeated strings
df = pd.read_csv('data.csv', dtype={'country': 'category'})

# 3. Parse dates during read (faster than after)
df = pd.read_csv('data.csv', parse_dates=['date_column'])

# 4. Use Parquet for large datasets (fastest read/write)
df.to_parquet('data.parquet')  # Much faster than CSV

# 5. Don't write index unless needed
df.to_csv('data.csv', index=False)

# 6. Use compression for large files
df.to_csv('data.csv.gz', compression='gzip', index=False)

# 7. Read only needed columns
df = pd.read_csv('data.csv', usecols=['col1', 'col2', 'col3'])

# 8. Use chunking for files larger than memory
for chunk in pd.read_csv('huge.csv', chunksize=50000):
    process_and_save(chunk)
```

---

## Data Inspection and Exploration

### Basic Information

```python
df = pd.read_csv('data.csv')

# Quick overview
print(df.head())          # First 5 rows
print(df.tail(10))        # Last 10 rows
print(df.sample(5))       # Random 5 rows

# Shape and size
print(df.shape)           # (rows, columns)
print(df.size)            # Total elements
print(len(df))            # Number of rows

# Column information
print(df.columns)         # Column names
print(df.columns.tolist())  # As list
print(df.dtypes)          # Data types per column
print(df.info())          # Comprehensive summary
print(df.info(memory_usage='deep'))  # Detailed memory usage

# Index information
print(df.index)
print(df.index.name)
```

### Statistical Summary

```python
# Numeric columns summary
print(df.describe())
# count, mean, std, min, 25%, 50%, 75%, max

# Include all columns (including non-numeric)
print(df.describe(include='all'))

# Specific statistics
print(df.describe(percentiles=[.1, .3, .5, .7, .9]))

# Object/categorical columns only
print(df.describe(include=['object', 'category']))

# Individual statistics
print(df.mean())          # Mean of numeric columns
print(df.median())        # Median
print(df.std())           # Standard deviation
print(df.var())           # Variance
print(df.min())           # Minimum
print(df.max())           # Maximum
print(df.sum())           # Sum
print(df.count())         # Non-null count
print(df.quantile(0.95))  # 95th percentile

# Per-column statistics
print(df['age'].mean())
print(df['salary'].median())
print(df['score'].std())

# Multiple aggregations
print(df.agg(['mean', 'std', 'min', 'max']))

# Different aggregations per column
print(df.agg({
    'age': ['mean', 'median'],
    'salary': ['min', 'max'],
    'score': 'std'
}))
```

### Unique Values and Counts

```python
# Unique values in a column
print(df['country'].unique())         # Array of unique values
print(df['country'].nunique())        # Count of unique values
print(df['country'].nunique(dropna=False))  # Include NaN in count

# Value counts (frequency)
print(df['country'].value_counts())            # Descending frequency
print(df['country'].value_counts(normalize=True))  # Proportions
print(df['country'].value_counts(dropna=False))    # Include NaN
print(df['country'].value_counts(ascending=True))  # Ascending order

# Top N values
print(df['country'].value_counts().head(10))

# Crosstab (frequency table for two columns)
print(pd.crosstab(df['gender'], df['country']))

# With margins (totals)
print(pd.crosstab(df['gender'], df['country'], margins=True))

# Normalize (show proportions)
print(pd.crosstab(df['gender'], df['country'], normalize='all'))
```

### Data Quality Checks

```python
# Check for missing values
print(df.isnull().sum())              # Count per column
print(df.isnull().sum() / len(df))    # Proportion per column
print(df.isna().any())                # True if any NaN in column

# Total missing values
print(df.isnull().sum().sum())

# Rows with any missing values
print(df[df.isnull().any(axis=1)])

# Visualize missing data pattern
import missingno as msno
msno.matrix(df)  # Requires missingno library

# Check for duplicates
print(df.duplicated().sum())          # Count duplicate rows
print(df[df.duplicated()])            # Show duplicate rows

# Duplicates based on specific columns
print(df.duplicated(subset=['id']).sum())
print(df[df.duplicated(subset=['email'], keep=False)])  # Show all duplicates

# Data type issues
print(df.dtypes)
print(df.select_dtypes(include='object').columns)  # Object columns

# Check for mixed types in columns
for col in df.select_dtypes(include='object').columns:
    types = df[col].apply(type).unique()
    if len(types) > 1:
        print(f"{col}: Mixed types {types}")

# Value range checks
print(df['age'].min(), df['age'].max())
print((df['age'] < 0).sum())  # Invalid ages
print((df['age'] > 120).sum())  # Suspicious ages

# Check for outliers (IQR method)
Q1 = df['salary'].quantile(0.25)
Q3 = df['salary'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['salary'] < Q1 - 1.5 * IQR) | (df['salary'] > Q3 + 1.5 * IQR)]
print(f"Outliers: {len(outliers)}")
```

### Memory Usage Analysis

```python
# Overall memory usage
print(df.memory_usage(deep=True))
print(f"Total: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

# Most memory-intensive columns
mem = df.memory_usage(deep=True)
print(mem.sort_values(ascending=False).head())

# Percentage per column
mem_pct = 100 * df.memory_usage(deep=True) / df.memory_usage(deep=True).sum()
print(mem_pct.sort_values(ascending=False))

# Info with memory details
df.info(memory_usage='deep')
```

### Quick Data Profiling

```python
# Comprehensive profile function
def profile_dataframe(df):
    """Generate comprehensive data profile"""
    print("=" * 50)
    print("DATAFRAME PROFILE")
    print("=" * 50)
    
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    
    print("\n" + "="*50)
    print("MISSING DATA")
    print("="*50)
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_df = pd.DataFrame({
        'Missing': missing,
        'Percent': missing_pct
    })
    print(missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False))
    
    print("\n" + "="*50)
    print("DUPLICATES")
    print("="*50)
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    print("\n" + "="*50)
    print("DATA TYPES")
    print("="*50)
    print(df.dtypes.value_counts())
    
    print("\n" + "="*50)
    print("NUMERIC SUMMARY")
    print("="*50)
    print(df.describe())
    
    print("\n" + "="*50)
    print("CATEGORICAL SUMMARY")
    print("="*50)
    for col in df.select_dtypes(include=['object', 'category']).columns:
        print(f"\n{col}: {df[col].nunique()} unique values")
        print(df[col].value_counts().head())

# Usage
profile_dataframe(df)
```

---

## Handling Missing Data

Missing data is inevitable in real-world datasets. Pandas provides comprehensive tools for detecting, analyzing, and handling missing values.

### Detecting Missing Data

```python
# Missing value representations in Pandas
# None, np.nan, pd.NA, pd.NaT (for datetimes)

import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, None, 7, 8],
    'C': ['a', 'b', 'c', None]
})

# Check for missing values
print(df.isnull())        # Boolean DataFrame
print(df.isna())          # Same as isnull()
print(df.notnull())       # Inverse of isnull()
print(df.notna())         # Same as notnull()

# Count missing values per column
print(df.isnull().sum())

# Count missing values per row
print(df.isnull().sum(axis=1))

# Total missing values
print(df.isnull().sum().sum())

# Percentage missing per column
print(100 * df.isnull().sum() / len(df))

# Columns with any missing values
print(df.columns[df.isnull().any()].tolist())

# Rows with any missing values
print(df[df.isnull().any(axis=1)])

# Rows with all missing values
print(df[df.isnull().all(axis=1)])

# Check specific column
print(df['A'].isnull())
print(df['A'].isnull().sum())
```

### Dropping Missing Data

```python
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, 7, 8],
    'C': [9, 10, 11, 12]
})

# Drop rows with ANY missing values
df_clean = df.dropna()
# Only row with index 3 remains

# Drop rows where ALL values are missing
df_clean = df.dropna(how='all')

# Drop columns with any missing values
df_clean = df.dropna(axis=1)
# Only column 'C' remains

# Drop rows with missing values in specific columns
df_clean = df.dropna(subset=['A', 'B'])

# Require at least N non-null values
df_clean = df.dropna(thresh=2)  # Keep rows with >= 2 non-null values

# In-place operation
df.dropna(inplace=True)

# Practical example: Remove rows where key columns are missing
df_clean = df.dropna(subset=['user_id', 'timestamp'])
```

### Filling Missing Data

```python
df = pd.DataFrame({
    'A': [1, np.nan, 3, np.nan, 5],
    'B': [10, 20, np.nan, 40, 50],
    'C': ['x', 'y', np.nan, 'z', 'w']
})

# Fill with scalar value
df_filled = df.fillna(0)
df_filled = df.fillna('missing')

# Fill with different values per column
df_filled = df.fillna({
    'A': 0,
    'B': df['B'].mean(),
    'C': 'unknown'
})

# Forward fill (propagate last valid observation)
df_filled = df.fillna(method='ffill')  # or method='pad'
# [1, 1, 3, 3, 5] for column A

# Backward fill (use next valid observation)
df_filled = df.fillna(method='bfill')  # or method='backfill'
# [1, 3, 3, 5, 5] for column A

# Limit forward/backward fill
df_filled = df.fillna(method='ffill', limit=1)  # Only fill 1 consecutive NaN

# Fill with column statistics
df['A'] = df['A'].fillna(df['A'].mean())
df['B'] = df['B'].fillna(df['B'].median())
df['C'] = df['C'].fillna(df['C'].mode()[0])  # Most frequent value

# Fill with group-specific values
df['score'] = df.groupby('category')['score'].transform(
    lambda x: x.fillna(x.mean())
)

# Interpolate (for numeric data)
df['A'] = df['A'].interpolate()              # Linear interpolation
df['A'] = df['A'].interpolate(method='polynomial', order=2)
df['A'] = df['A'].interpolate(method='spline', order=3)

# Time series interpolation
df['value'] = df['value'].interpolate(method='time')

# In-place filling
df.fillna(0, inplace=True)
```

### Replacing Values

```python
df = pd.DataFrame({
    'A': [1, 2, -999, 4, -999],
    'B': ['yes', 'no', 'unknown', 'yes', 'no'],
    'C': [10.0, 20.0, 30.0, -1.0, 50.0]
})

# Replace specific value
df_clean = df.replace(-999, np.nan)

# Replace multiple values
df_clean = df.replace([-999, -1], np.nan)

# Replace with different values
df_clean = df.replace({-999: np.nan, -1: 0})

# Replace per column
df_clean = df.replace({
    'A': {-999: np.nan},
    'B': {'unknown': np.nan},
    'C': {-1: 0}
})

# Replace using regex
df['B'] = df['B'].replace(r'unkn.*', np.nan, regex=True)

# Replace in-place
df.replace(-999, np.nan, inplace=True)

# Chain replace operations
df_clean = (df
    .replace(-999, np.nan)
    .replace('unknown', np.nan)
    .fillna(method='ffill'))
```

### Advanced Missing Data Handling

```python
# Create missing data indicator columns
df['A_missing'] = df['A'].isnull().astype(int)

# Fill missing with prediction from other columns
from sklearn.impute import SimpleImputer, KNNImputer

# Mean/median/mode imputation
imputer = SimpleImputer(strategy='mean')
df[['A', 'B']] = imputer.fit_transform(df[['A', 'B']])

# KNN imputation (uses similar rows)
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)

# Multiple imputation (MICE-like approach)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(random_state=42)
df_imputed = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)

# Conditional filling
df.loc[df['category'] == 'A', 'value'] = df.loc[df['category'] == 'A', 'value'].fillna(0)

# Fill with rolling statistics
df['value'] = df['value'].fillna(df['value'].rolling(window=3, min_periods=1).mean())
```

### Best Practices for Missing Data

```python
# 1. Always investigate WHY data is missing
missing_analysis = df.isnull().sum()
print(missing_analysis[missing_analysis > 0])

# 2. Document your missing data strategy
def handle_missing_data(df):
    """
    Missing data handling strategy:
    - user_id: Drop rows (critical field)
    - age: Fill with median
    - income: Fill with group median
    - comments: Fill with empty string
    """
    df = df.dropna(subset=['user_id'])
    df['age'] = df['age'].fillna(df['age'].median())
    df['income'] = df.groupby('region')['income'].transform(
        lambda x: x.fillna(x.median())
    )
    df['comments'] = df['comments'].fillna('')
    return df

# 3. Consider the impact on statistics
print("Before dropping NaN:")
print(f"Mean: {df['value'].mean()}")
print(f"Count: {df['value'].count()}")

df_clean = df.dropna(subset=['value'])
print("\nAfter dropping NaN:")
print(f"Mean: {df_clean['value'].mean()}")
print(f"Count: {df_clean['value'].count()}")

# 4. Create flags for missing data (useful for ML)
df['value_was_missing'] = df['value'].isnull().astype(int)
df['value'] = df['value'].fillna(df['value'].median())

# 5. Use appropriate method based on data type
# Numeric: mean, median, interpolation
# Categorical: mode, separate "missing" category
# Time series: forward fill, interpolation
# Critical fields: drop rows

# 6. Avoid data leakage in train/test splits
# Calculate fill values ONLY from training data
train_mean = train_df['value'].mean()
train_df['value'] = train_df['value'].fillna(train_mean)
test_df['value'] = test_df['value'].fillna(train_mean)  # Use train mean!
```

---

## Data Manipulation and Transformation

### Sorting

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 25, 35],
    'salary': [50000, 60000, 55000, 70000]
})

# Sort by single column
df_sorted = df.sort_values('age')
df_sorted = df.sort_values('age', ascending=False)  # Descending

# Sort by multiple columns
df_sorted = df.sort_values(['age', 'salary'])  # Age first, then salary
df_sorted = df.sort_values(['age', 'salary'], ascending=[True, False])

# Sort by index
df_sorted = df.sort_index()
df_sorted = df.sort_index(ascending=False)

# In-place sorting
df.sort_values('age', inplace=True)

# Handle NaN in sorting
df_sorted = df.sort_values('age', na_position='first')  # NaN at start
df_sorted = df.sort_values('age', na_position='last')   # NaN at end (default)

# Sort by custom key function
df_sorted = df.sort_values('name', key=lambda x: x.str.lower())

# Sort columns (not rows)
df_sorted = df.sort_index(axis=1)  # Alphabetically by column names

# Get sorted indices
sorted_indices = df['age'].argsort()
```

### Filtering

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 40, 28],
    'city': ['NY', 'LA', 'NY', 'SF', 'LA'],
    'salary': [50000, 60000, 70000, 80000, 55000]
})

# Single condition
high_earners = df[df['salary'] > 60000]
young_people = df[df['age'] < 30]

# Multiple conditions (AND)
young_high_earners = df[(df['age'] < 35) & (df['salary'] > 55000)]

# Multiple conditions (OR)
ny_or_sf = df[(df['city'] == 'NY') | (df['city'] == 'SF')]

# NOT condition
not_la = df[~(df['city'] == 'LA')]

# Using isin() for multiple values
ny_la = df[df['city'].isin(['NY', 'LA'])]

# String contains
names_with_a = df[df['name'].str.contains('a', case=False)]

# Between (inclusive)
mid_age = df[df['age'].between(28, 35)]

# Query method (SQL-like syntax)
result = df.query('age > 30 and salary < 75000')
result = df.query('city in ["NY", "LA"]')
result = df.query('age > @threshold')  # Use variable with @

# Filter by index
df_subset = df[df.index.isin([0, 2, 4])]

# Filter columns by dtype
numeric_cols = df.select_dtypes(include=['int64', 'float64'])
object_cols = df.select_dtypes(include='object')
numeric_cols = df.select_dtypes(include='number')  # All numeric types

# Filter columns by name pattern
cols_with_age = df.filter(like='age')
cols_starting_with_s = df.filter(regex='^s')
```

### Adding and Modifying Columns

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'salary': [50000, 60000, 70000]
})

# Add new column with scalar
df['bonus'] = 5000

# Add column from calculation
df['total_comp'] = df['salary'] + df['bonus']

# Add column with conditional logic
df['high_earner'] = df['salary'] > 60000

# Add column using numpy where
df['category'] = np.where(df['salary'] > 60000, 'High', 'Low')

# Add column using multiple conditions
conditions = [
    df['salary'] < 55000,
    (df['salary'] >= 55000) & (df['salary'] < 65000),
    df['salary'] >= 65000
]
choices = ['Low', 'Medium', 'High']
df['salary_band'] = np.select(conditions, choices, default='Unknown')

# Add column using apply
df['name_length'] = df['name'].apply(len)
df['upper_name'] = df['name'].apply(lambda x: x.upper())

# Add column using map (dictionary)
mapping = {'Alice': 'Team A', 'Bob': 'Team B', 'Charlie': 'Team A'}
df['team'] = df['name'].map(mapping)

# Add column from another DataFrame
df['department'] = other_df['department']

# Insert column at specific position
df.insert(1, 'employee_id', [101, 102, 103])

# Modify existing column
df['salary'] = df['salary'] * 1.1  # 10% raise
df['name'] = df['name'].str.upper()

# Modify with loc
df.loc[df['salary'] > 60000, 'bonus'] = 10000

# Modify multiple columns
df[['salary', 'bonus']] = df[['salary', 'bonus']] * 1.05
```

### Removing Columns and Rows

```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})

# Drop single column
df_dropped = df.drop('A', axis=1)
df_dropped = df.drop(columns='A')  # More explicit

# Drop multiple columns
df_dropped = df.drop(['A', 'B'], axis=1)
df_dropped = df.drop(columns=['A', 'B'])

# Drop rows
df_dropped = df.drop(0, axis=0)  # Drop row at index 0
df_dropped = df.drop([0, 2])     # Drop multiple rows
df_dropped = df.drop(index=[0, 2])

# Drop in-place
df.drop('A', axis=1, inplace=True)

# Drop with condition
df_dropped = df.drop(df[df['A'] < 2].index)

# Drop columns by condition
cols_to_drop = [col for col in df.columns if 'temp' in col]
df_dropped = df.drop(columns=cols_to_drop)

# Drop duplicates
df_unique = df.drop_duplicates()
df_unique = df.drop_duplicates(subset=['A'])  # Based on column A only
df_unique = df.drop_duplicates(subset=['A', 'B'], keep='last')  # Keep last occurrence
df_unique = df.drop_duplicates(keep=False)  # Drop all duplicates

# Remove rows with missing values
df_clean = df.dropna()
```

### Renaming

```python
df = pd.DataFrame({
    'old_name1': [1, 2, 3],
    'old_name2': [4, 5, 6]
})

# Rename specific columns
df_renamed = df.rename(columns={'old_name1': 'new_name1'})

# Rename multiple columns
df_renamed = df.rename(columns={
    'old_name1': 'new_name1',
    'old_name2': 'new_name2'
})

# Rename using function
df_renamed = df.rename(columns=str.upper)  # Uppercase all column names
df_renamed = df.rename(columns=lambda x: x.replace('old', 'new'))

# Rename index
df_renamed = df.rename(index={0: 'first', 1: 'second'})

# Rename in-place
df.rename(columns={'old_name1': 'new_name1'}, inplace=True)

# Set column names directly
df.columns = ['col1', 'col2']

# Clean column names (remove spaces, lowercase)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Rename index name
df.index.name = 'row_id'
df.columns.name = 'features'
```

### Apply and Transform

```python
df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [10, 20, 30, 40]
})

# Apply function to column (returns Series)
df['A_squared'] = df['A'].apply(lambda x: x ** 2)

# Apply function to multiple columns
df['sum'] = df.apply(lambda row: row['A'] + row['B'], axis=1)

# Apply with additional arguments
def multiply(x, factor):
    return x * factor

df['A_times_3'] = df['A'].apply(multiply, factor=3)

# Apply to all elements (deprecated, use map instead)
# df_transformed = df.applymap(lambda x: x * 2)  # Old way
df_transformed = df.map(lambda x: x * 2)  # New way

# Transform (keeps same shape)
df['A_normalized'] = df.groupby('category')['A'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Multiple functions with agg
result = df.agg(['sum', 'mean', 'std'])

# Different functions per column
result = df.agg({
    'A': ['sum', 'mean'],
    'B': ['min', 'max']
})

# Custom aggregation function
def custom_agg(x):
    return x.max() - x.min()

result = df.agg(custom_agg)

# Apply with multiple outputs (returns DataFrame)
def multiple_outputs(x):
    return pd.Series({
        'min': x.min(),
        'max': x.max(),
        'range': x.max() - x.min()
    })

result = df.apply(multiple_outputs)
```

### Binning and Discretization

```python
df = pd.DataFrame({
    'age': [25, 35, 45, 55, 65, 75],
    'salary': [30000, 45000, 60000, 75000, 90000, 105000]
})

# Cut into equal-width bins
df['age_group'] = pd.cut(df['age'], bins=3)
# Output: (24.95, 41.667], (41.667, 58.333], (58.333, 75.0]

# Specify bin edges
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 100])

# Custom labels
df['age_group'] = pd.cut(df['age'], 
                         bins=[0, 30, 50, 100],
                         labels=['Young', 'Middle', 'Senior'])

# Include left edge instead of right
df['age_group'] = pd.cut(df['age'], bins=3, right=False)

# Cut into equal-frequency bins (quantiles)
df['salary_quartile'] = pd.qcut(df['salary'], q=4)

# Quantiles with labels
df['salary_quartile'] = pd.qcut(df['salary'], 
                                q=4, 
                                labels=['Q1', 'Q2', 'Q3', 'Q4'])

# Specific quantiles
df['salary_group'] = pd.qcut(df['salary'], 
                             q=[0, 0.25, 0.75, 1.0],
                             labels=['Low', 'Medium', 'High'])

# Get bin edges and categories
df['age_binned'], bins = pd.cut(df['age'], bins=3, retbins=True)
print(f"Bin edges: {bins}")
```

### Reshaping Data

```python
# Wide to long format (melt/unpivot)
df_wide = pd.DataFrame({
    'id': [1, 2, 3],
    'Jan': [100, 150, 200],
    'Feb': [110, 160, 210],
    'Mar': [120, 170, 220]
})

df_long = pd.melt(df_wide, 
                  id_vars=['id'],
                  value_vars=['Jan', 'Feb', 'Mar'],
                  var_name='month',
                  value_name='sales')
# Result:
#    id month  sales
# 0   1   Jan    100
# 1   2   Jan    150
# ...

# Long to wide format (pivot)
df_wide = df_long.pivot(index='id', columns='month', values='sales')

# Pivot with multiple values
df_wide = df_long.pivot(index='id', columns='month', values=['sales', 'quantity'])

# Pivot table (with aggregation)
df_pivot = pd.pivot_table(df,
                          values='sales',
                          index='region',
                          columns='product',
                          aggfunc='sum',
                          fill_value=0,
                          margins=True)  # Add totals

# Stack (pivot columns to rows)
df_stacked = df.stack()

# Unstack (pivot rows to columns)
df_unstacked = df.unstack()

# Unstack specific level (for MultiIndex)
df_unstacked = df.unstack(level=0)

# Transpose
df_transposed = df.T
```

### Index Operations

```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
}, index=['x', 'y', 'z'])

# Set column as index
df_indexed = df.set_index('A')

# Set multiple columns as index (MultiIndex)
df_indexed = df.set_index(['A', 'B'])

# Reset index (move index to column)
df_reset = df.reset_index()

# Reset specific level of MultiIndex
df_reset = df.reset_index(level=0)

# Drop index when resetting
df_reset = df.reset_index(drop=True)

# Rename index
df.index = ['row1', 'row2', 'row3']
df.index.name = 'row_id'

# Reindex (change order or add missing indices)
df_reindexed = df.reindex(['x', 'z', 'w'])  # 'w' will have NaN

# Reindex with fill
df_reindexed = df.reindex(['x', 'z', 'w'], fill_value=0)

# Sort index
df_sorted = df.sort_index()

# Check if index is unique
print(df.index.is_unique)

# Remove duplicates from index
df = df[~df.index.duplicated(keep='first')]
```

---

## Grouping and Aggregation

The **GroupBy** operation is one of Pandas' most powerful features, implementing the "split-apply-combine" pattern for data analysis.

### Basic GroupBy

```python
df = pd.DataFrame({
    'category': ['A', 'B', 'A', 'B', 'A', 'B'],
    'region': ['East', 'East', 'West', 'West', 'East', 'West'],
    'sales': [100, 150, 200, 120, 130, 180],
    'quantity': [10, 15, 20, 12, 13, 18]
})

# Group by single column
grouped = df.groupby('category')

# Access groups
print(grouped.groups)  # Dict of group labels to indices
print(grouped.get_group('A'))  # Get specific group

# Iterate over groups
for name, group in grouped:
    print(f"Group: {name}")
    print(group)

# Group by multiple columns
grouped = df.groupby(['category', 'region'])

# Group by index level (for MultiIndex)
grouped = df.groupby(level=0)
```

### Aggregation Functions

```python
# Single aggregation
print(df.groupby('category')['sales'].sum())
print(df.groupby('category')['sales'].mean())
print(df.groupby('category')['sales'].count())
print(df.groupby('category')['sales'].std())

# Multiple aggregations
print(df.groupby('category').agg(['sum', 'mean', 'count']))

# Different aggregations per column
result = df.groupby('category').agg({
    'sales': ['sum', 'mean'],
    'quantity': ['min', 'max']
})

# Named aggregations (cleaner column names)
result = df.groupby('category').agg(
    total_sales=('sales', 'sum'),
    avg_sales=('sales', 'mean'),
    max_quantity=('quantity', 'max')
)

# Custom aggregation function
def range_func(x):
    return x.max() - x.min()

result = df.groupby('category')['sales'].agg(range_func)

# Multiple custom functions
result = df.groupby('category')['sales'].agg([
    'sum',
    ('range', range_func),
    ('cv', lambda x: x.std() / x.mean())  # Coefficient of variation
])

# Aggregate all numeric columns
result = df.groupby('category').sum()
result = df.groupby('category').mean()
```

### Transform and Filter

```python
# Transform (returns same shape as input)
# Calculate group mean and add as new column
df['category_mean'] = df.groupby('category')['sales'].transform('mean')

# Z-score normalization within groups
df['sales_normalized'] = df.groupby('category')['sales'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Percentage of group total
df['pct_of_group'] = df.groupby('category')['sales'].transform(
    lambda x: 100 * x / x.sum()
)

# Filter (keep only groups that meet condition)
# Keep groups where total sales > 400
df_filtered = df.groupby('category').filter(lambda x: x['sales'].sum() > 400)

# Keep groups with more than 2 members
df_filtered = df.groupby('category').filter(lambda x: len(x) > 2)

# Keep top N groups by sum
top_categories = df.groupby('category')['sales'].sum().nlargest(3).index
df_filtered = df[df['category'].isin(top_categories)]
```

### Apply (Most Flexible)

```python
# Apply custom function to each group
def top_sales(group):
    """Return row with highest sales in group"""
    return group.nlargest(1, 'sales')

result = df.groupby('category').apply(top_sales)

# Multiple operations in one function
def group_stats(group):
    """Calculate custom statistics for group"""
    return pd.Series({
        'total_sales': group['sales'].sum(),
        'avg_sales': group['sales'].mean(),
        'num_transactions': len(group),
        'max_quantity': group['quantity'].max()
    })

result = df.groupby('category').apply(group_stats)

# Apply with additional arguments
def scale_sales(group, factor):
    group['scaled_sales'] = group['sales'] * factor
    return group

result = df.groupby('category').apply(scale_sales, factor=1.1)
```

### Advanced GroupBy Operations

```python
# Size (count of group members, includes NaN)
print(df.groupby('category').size())

# Count (excludes NaN)
print(df.groupby('category').count())

# First and last values
print(df.groupby('category').first())
print(df.groupby('category').last())

# Nth value
print(df.groupby('category').nth(0))  # First row
print(df.groupby('category').nth(-1))  # Last row

# Head and tail per group
print(df.groupby('category').head(2))  # First 2 rows per group
print(df.groupby('category').tail(2))  # Last 2 rows per group

# Cumulative operations within groups
df['cumsum_sales'] = df.groupby('category')['sales'].cumsum()
df['cummax_sales'] = df.groupby('category')['sales'].cummax()
df['rank_in_group'] = df.groupby('category')['sales'].rank()

# Shift within groups
df['prev_sales'] = df.groupby('category')['sales'].shift(1)
df['next_sales'] = df.groupby('category')['sales'].shift(-1)

# Percent change within groups
df['sales_pct_change'] = df.groupby('category')['sales'].pct_change()

# Rolling operations within groups
df['rolling_mean'] = df.groupby('category')['sales'].rolling(window=2).mean().reset_index(0, drop=True)
```

### Multiple Group Keys

```python
# Group by multiple columns
grouped = df.groupby(['category', 'region'])

# Aggregate
result = grouped['sales'].sum()
# Returns Series with MultiIndex

# Convert to DataFrame
result = grouped['sales'].sum().reset_index()

# Unstack to pivot
result = grouped['sales'].sum().unstack()
#         East  West
# A        230   200
# B        150   300

# Group by combination of column and calculated value
df['month'] = pd.to_datetime(df['date']).dt.month
grouped = df.groupby(['category', df['date'].dt.year])
```

### GroupBy with Missing Values

```python
df = pd.DataFrame({
    'category': ['A', 'B', np.nan, 'A', 'B'],
    'value': [1, 2, 3, 4, 5]
})

# By default, NaN is excluded from groups
grouped = df.groupby('category')  # Only 'A' and 'B' groups

# Include NaN as a group
grouped = df.groupby('category', dropna=False)  # 'A', 'B', and NaN groups
```

### Practical GroupBy Examples

```python
# Example: Customer analytics
orders = pd.DataFrame({
    'customer_id': [1, 1, 2, 2, 3, 3, 3],
    'order_date': pd.date_range('2024-01-01', periods=7),
    'amount': [100, 150, 200, 250, 50, 75, 100],
    'product': ['A', 'B', 'A', 'C', 'A', 'A', 'B']
})

# Customer lifetime value
clv = orders.groupby('customer_id').agg({
    'amount': 'sum',
    'order_date': 'count'
}).rename(columns={'amount': 'total_spent', 'order_date': 'num_orders'})

# Average order value per customer
clv['avg_order_value'] = clv['total_spent'] / clv['num_orders']

# Most purchased product per customer
most_purchased = orders.groupby(['customer_id', 'product']).size().reset_index(name='count')
most_purchased = most_purchased.loc[most_purchased.groupby('customer_id')['count'].idxmax()]

# Days since last purchase
orders['days_since_last'] = (
    orders.groupby('customer_id')['order_date']
    .transform('max')
    .apply(lambda x: (pd.Timestamp.now() - x).days)
)

# Cohort analysis
orders['cohort'] = orders.groupby('customer_id')['order_date'].transform('min')
orders['cohort_month'] = orders['cohort'].dt.to_period('M')
orders['order_month'] = orders['order_date'].dt.to_period('M')

cohort_data = orders.groupby(['cohort_month', 'order_month']).agg({
    'customer_id': 'nunique',
    'amount': 'sum'
})
```

---

## Merging, Joining, and Concatenation

Combining DataFrames is essential for integrating data from multiple sources.

### Concatenation

```python
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
df3 = pd.DataFrame({'C': [9, 10], 'D': [11, 12]})

# Vertical concatenation (stack rows)
result = pd.concat([df1, df2])
# Index: 0, 1, 0, 1 (duplicated)

# Reset index after concatenation
result = pd.concat([df1, df2], ignore_index=True)
# Index: 0, 1, 2, 3

# Horizontal concatenation (stack columns)
result = pd.concat([df1, df3], axis=1)

# Concatenate with keys (create MultiIndex)
result = pd.concat([df1, df2], keys=['first', 'second'])

# Inner join (only common columns)
df4 = pd.DataFrame({'A': [1, 2], 'C': [3, 4]})
result = pd.concat([df1, df4], join='inner')  # Only column 'A'

# Outer join (all columns, default)
result = pd.concat([df1, df4], join='outer')  # Columns 'A', 'B', 'C' with NaN

# Append (simplified concat for rows, deprecated in favor of concat)
# result = df1.append(df2)  # Old way
result = pd.concat([df1, df2])  # Preferred way
```

### Merge (SQL-style Joins)

```python
left = pd.DataFrame({
    'key': ['A', 'B', 'C', 'D'],
    'value_left': [1, 2, 3, 4]
})

right = pd.DataFrame({
    'key': ['B', 'C', 'D', 'E'],
    'value_right': [5, 6, 7, 8]
})

# Inner join (intersection, default)
result = pd.merge(left, right, on='key')
# key  value_left  value_right
#  B           2            5
#  C           3            6
#  D           4            7

# Left join (all from left, matching from right)
result = pd.merge(left, right, on='key', how='left')
# Includes A with NaN for value_right

# Right join (all from right, matching from left)
result = pd.merge(left, right, on='key', how='right')
# Includes E with NaN for value_left

# Outer join (all records from both)
result = pd.merge(left, right, on='key', how='outer')
# Includes A and E with NaN where no match

# Merge on multiple columns
result = pd.merge(left, right, on=['key1', 'key2'])

# Merge with different column names
result = pd.merge(left, right, left_on='key_left', right_on='key_right')

# Merge on index
result = pd.merge(left, right, left_index=True, right_index=True)

# Suffix for overlapping column names
result = pd.merge(left, right, on='key', suffixes=('_l', '_r'))

# Indicator column (shows source of each row)
result = pd.merge(left, right, on='key', how='outer', indicator=True)
# _merge column: 'left_only', 'right_only', or 'both'

# Validate merge (check for duplicates)
result = pd.merge(left, right, on='key', validate='one_to_one')
# Options: 'one_to_one', 'one_to_many', 'many_to_one', 'many_to_many'
```

### Join (Index-based Merge)

```python
left = pd.DataFrame({
    'value_left': [1, 2, 3]
}, index=['A', 'B', 'C'])

right = pd.DataFrame({
    'value_right': [4, 5, 6]
}, index=['B', 'C', 'D'])

# Join on index (left join by default)
result = left.join(right)

# Specify join type
result = left.join(right, how='inner')
result = left.join(right, how='outer')

# Join multiple DataFrames
result = left.join([right, another_df])

# Join on specific column from right
result = left.join(right.set_index('key'), on='key')
```

### Merge Comparison Examples

```python
# SQL equivalents

# INNER JOIN
# SELECT * FROM left INNER JOIN right ON left.key = right.key
result = pd.merge(left, right, on='key', how='inner')

# LEFT JOIN
# SELECT * FROM left LEFT JOIN right ON left.key = right.key
result = pd.merge(left, right, on='key', how='left')

# RIGHT JOIN
# SELECT * FROM left RIGHT JOIN right ON left.key = right.key
result = pd.merge(left, right, on='key', how='right')

# FULL OUTER JOIN
# SELECT * FROM left FULL OUTER JOIN right ON left.key = right.key
result = pd.merge(left, right, on='key', how='outer')

# Multiple join keys
# SELECT * FROM left JOIN right ON left.k1 = right.k1 AND left.k2 = right.k2
result = pd.merge(left, right, on=['k1', 'k2'])
```

### Handling Duplicate Keys

```python
left = pd.DataFrame({
    'key': ['A', 'B', 'B'],
    'value': [1, 2, 3]
})

right = pd.DataFrame({
    'key': ['B', 'B', 'C'],
    'value': [4, 5, 6]
})

# Many-to-many merge (Cartesian product for duplicates)
result = pd.merge(left, right, on='key')
# Creates 4 rows for key 'B' (2 from left × 2 from right)

# Validate to catch unexpected duplicates
try:
    result = pd.merge(left, right, on='key', validate='one_to_one')
except pd.errors.MergeError as e:
    print(f"Merge validation failed: {e}")
```

### Performance Tips

```python
# 1. Sort and merge (faster for large sorted data)
left_sorted = left.sort_values('key')
right_sorted = right.sort_values('key')
result = pd.merge(left_sorted, right_sorted, on='key')

# 2. Merge on index (faster than column merge)
left_indexed = left.set_index('key')
right_indexed = right.set_index('key')
result = left_indexed.join(right_indexed)

# 3. Use categorical dtype for merge keys (reduces memory)
left['key'] = left['key'].astype('category')
right['key'] = right['key'].astype('category')
result = pd.merge(left, right, on='key')

# 4. Filter before merging
# Don't merge entire DataFrames if you only need subset
left_filtered = left[left['date'] > '2024-01-01']
result = pd.merge(left_filtered, right, on='key')

# 5. Use copy=False to avoid unnecessary copies
result = pd.merge(left, right, on='key', copy=False)
```

---

## Time Series Operations

Pandas has extensive support for time series data, making it ideal for financial analysis, IoT data, and temporal analytics.

### Creating DateTime Objects

```python
# Parse strings to datetime
dates = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
print(dates)

# Parse with custom format
dates = pd.to_datetime(['01/15/2024', '02/20/2024'], format='%m/%d/%Y')

# Handle parsing errors
dates = pd.to_datetime(['2024-01-01', 'invalid'], errors='coerce')  # NaT for invalid
dates = pd.to_datetime(['2024-01-01', 'invalid'], errors='ignore')  # Keep as string

# Convert Unix timestamp
timestamp = pd.to_datetime(1609459200, unit='s')

# From components
dates = pd.to_datetime({
    'year': [2024, 2024],
    'month': [1, 2],
    'day': [15, 20]
})

# Current timestamp
now = pd.Timestamp.now()
today = pd.Timestamp.today()

# Specific timestamp
ts = pd.Timestamp('2024-01-15 14:30:00')
ts = pd.Timestamp(year=2024, month=1, day=15, hour=14, minute=30)
```

### Date Ranges

```python
# Daily date range
dates = pd.date_range(start='2024-01-01', end='2024-01-10')
dates = pd.date_range(start='2024-01-01', periods=10)  # 10 days

# Specific frequency
dates = pd.date_range('2024-01-01', periods=10, freq='D')    # Daily
dates = pd.date_range('2024-01-01', periods=10, freq='W')    # Weekly
dates = pd.date_range('2024-01-01', periods=12, freq='M')    # Month end
dates = pd.date_range('2024-01-01', periods=12, freq='MS')   # Month start
dates = pd.date_range('2024-01-01', periods=4, freq='Q')     # Quarter end
dates = pd.date_range('2024-01-01', periods=5, freq='Y')     # Year end
dates = pd.date_range('2024-01-01', periods=24, freq='H')    # Hourly
dates = pd.date_range('2024-01-01', periods=60, freq='T')    # Minute
dates = pd.date_range('2024-01-01', periods=60, freq='S')    # Second

# Business days only
dates = pd.bdate_range('2024-01-01', periods=10)  # Excludes weekends

# Custom business days (exclude specific days)
from pandas.tseries.offsets import CustomBusinessDay
cbd = CustomBusinessDay(holidays=['2024-01-15', '2024-02-20'])
dates = pd.bdate_range('2024-01-01', periods=10, freq=cbd)

# Time deltas
time_delta = pd.timedelta_range(start='1 day', periods=10)
time_delta = pd.timedelta_range(start='1 hour', periods=24, freq='H')
```

### DateTime Components and Attributes

```python
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=100, freq='D'),
    'value': np.random.randn(100)
})

# Set datetime as index
df = df.set_index('date')

# Extract components using .dt accessor
df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
df['dayofweek'] = df.index.dayofweek  # Monday=0, Sunday=6
df['dayofyear'] = df.index.dayofyear
df['week'] = df.index.isocalendar().week
df['quarter'] = df.index.quarter
df['hour'] = df.index.hour
df['minute'] = df.index.minute

# Date properties
df['is_month_start'] = df.index.is_month_start
df['is_month_end'] = df.index.is_month_end
df['is_quarter_start'] = df.index.is_quarter_start
df['is_quarter_end'] = df.index.is_quarter_end
df['is_year_start'] = df.index.is_year_start
df['is_year_end'] = df.index.is_year_end
df['is_leap_year'] = df.index.is_leap_year

# Day name
df['day_name'] = df.index.day_name()  # 'Monday', 'Tuesday', etc.
df['month_name'] = df.index.month_name()  # 'January', 'February', etc.

# For datetime column (not index)
df['date_col'] = pd.date_range('2024-01-01', periods=100)
df['year'] = df['date_col'].dt.year
df['month'] = df['date_col'].dt.month
df['day_name'] = df['date_col'].dt.day_name()
```

### DateTime Indexing and Slicing

```python
df = pd.DataFrame({
    'value': np.random.randn(365)
}, index=pd.date_range('2024-01-01', periods=365))

# Select by date string
df_jan = df['2024-01']          # All of January 2024
df_q1 = df['2024-01':'2024-03'] # Q1 2024
df_day = df['2024-01-15']       # Specific day

# Select by year
df_2024 = df['2024']

# Partial string indexing
df_jan_15 = df['2024-01-15']

# Using loc with dates
df_range = df.loc['2024-01-01':'2024-01-31']

# Boolean indexing with dates
df_weekends = df[df.index.dayofweek >= 5]
df_january = df[df.index.month == 1]
df_q1 = df[df.index.quarter == 1]

# Between dates
mask = (df.index >= '2024-03-01') & (df.index <= '2024-03-31')
df_march = df[mask]

# Using between
df_march = df[df.index.to_series().between('2024-03-01', '2024-03-31')]
```

### Resampling (Frequency Conversion)

```python
# Downsample: high frequency to low frequency
df = pd.DataFrame({
    'value': np.random.randn(365)
}, index=pd.date_range('2024-01-01', periods=365, freq='D'))

# Daily to weekly (sum)
df_weekly = df.resample('W').sum()

# Daily to monthly (mean)
df_monthly = df.resample('M').mean()

# Multiple aggregations
df_monthly = df.resample('M').agg(['sum', 'mean', 'std', 'min', 'max'])

# Different aggregation per column
df_monthly = df.resample('M').agg({
    'sales': 'sum',
    'temperature': 'mean'
})

# Resample with custom function
df_monthly = df.resample('M').apply(lambda x: x.max() - x.min())

# Label and closed side
df_weekly = df.resample('W', label='left').sum()  # Use start of period
df_weekly = df.resample('W', closed='left').sum()  # Which side is closed

# Upsample: low frequency to high frequency
df_daily = df_monthly.resample('D').ffill()  # Forward fill
df_daily = df_monthly.resample('D').bfill()  # Backward fill
df_daily = df_monthly.resample('D').interpolate()  # Interpolate

# Combine downsampling and upsampling
df_6h = df.resample('6H').mean()  # Daily to 6-hourly
```

### Rolling Windows

```python
df = pd.DataFrame({
    'value': np.random.randn(100)
}, index=pd.date_range('2024-01-01', periods=100))

# Simple moving average
df['SMA_7'] = df['value'].rolling(window=7).mean()

# Multiple window sizes
df['SMA_7'] = df['value'].rolling(7).mean()
df['SMA_30'] = df['value'].rolling(30).mean()

# Other rolling aggregations
df['rolling_sum'] = df['value'].rolling(7).sum()
df['rolling_std'] = df['value'].rolling(7).std()
df['rolling_min'] = df['value'].rolling(7).min()
df['rolling_max'] = df['value'].rolling(7).max()

# Custom rolling function
df['rolling_range'] = df['value'].rolling(7).apply(lambda x: x.max() - x.min())

# Minimum periods (don't wait for full window)
df['SMA'] = df['value'].rolling(window=7, min_periods=1).mean()

# Center the window (default is right-aligned)
df['centered_MA'] = df['value'].rolling(window=7, center=True).mean()

# Weighted moving average
weights = np.arange(1, 8)
df['WMA'] = df['value'].rolling(7).apply(lambda x: np.dot(x, weights) / weights.sum())

# Exponential moving average
df['EMA'] = df['value'].ewm(span=7, adjust=False).mean()

# Bollinger Bands
window = 20
df['MA'] = df['value'].rolling(window).mean()
df['STD'] = df['value'].rolling(window).std()
df['upper_band'] = df['MA'] + (2 * df['STD'])
df['lower_band'] = df['MA'] - (2 * df['STD'])
```

### Expanding Windows

```python
# Cumulative statistics (expanding from start)
df['cumulative_sum'] = df['value'].expanding().sum()
df['cumulative_mean'] = df['value'].expanding().mean()
df['cumulative_max'] = df['value'].expanding().max()
df['cumulative_min'] = df['value'].expanding().min()

# Expanding with minimum periods
df['expanding_mean'] = df['value'].expanding(min_periods=10).mean()
```

### Shifting and Lagging

```python
df = pd.DataFrame({
    'value': [1, 2, 3, 4, 5]
}, index=pd.date_range('2024-01-01', periods=5))

# Shift values (lag)
df['lag_1'] = df['value'].shift(1)   # Previous value
df['lag_2'] = df['value'].shift(2)   # 2 periods ago

# Shift forward (lead)
df['lead_1'] = df['value'].shift(-1)  # Next value

# Shift by time period (not index position)
df['lag_1day'] = df['value'].shift(1, freq='D')

# Calculate change
df['diff'] = df['value'].diff()         # value[t] - value[t-1]
df['pct_change'] = df['value'].pct_change()  # Percentage change

# Lag multiple periods at once
for i in range(1, 4):
    df[f'lag_{i}'] = df['value'].shift(i)
```

### Time Zones

```python
# Create timezone-aware datetime
dates = pd.date_range('2024-01-01', periods=5, tz='UTC')

# Localize (add timezone to naive datetime)
df = pd.DataFrame({
    'value': [1, 2, 3, 4, 5]
}, index=pd.date_range('2024-01-01', periods=5))

df.index = df.index.tz_localize('UTC')

# Convert timezone
df.index = df.index.tz_convert('US/Eastern')
df.index = df.index.tz_convert('Asia/Tokyo')

# Remove timezone
df.index = df.index.tz_localize(None)

# For datetime column
df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].dt.tz_localize('UTC')
df['date'] = df['date'].dt.tz_convert('US/Pacific')
```

### Practical Time Series Examples

```python
# Example: Stock price analysis
stock_data = pd.DataFrame({
    'price': np.random.randn(252).cumsum() + 100,
    'volume': np.random.randint(1000, 10000, 252)
}, index=pd.date_range('2024-01-01', periods=252, freq='B'))

# Daily returns
stock_data['returns'] = stock_data['price'].pct_change()

# Volatility (rolling standard deviation of returns)
stock_data['volatility'] = stock_data['returns'].rolling(21).std() * np.sqrt(252)

# Moving averages
stock_data['SMA_50'] = stock_data['price'].rolling(50).mean()
stock_data['SMA_200'] = stock_data['price'].rolling(200).mean()

# Trading signals (Golden Cross / Death Cross)
stock_data['signal'] = 0
stock_data.loc[stock_data['SMA_50'] > stock_data['SMA_200'], 'signal'] = 1  # Buy
stock_data.loc[stock_data['SMA_50'] < stock_data['SMA_200'], 'signal'] = -1  # Sell

# Monthly aggregation
monthly = stock_data.resample('M').agg({
    'price': 'last',      # Last price of month
    'volume': 'sum',      # Total volume
    'returns': 'sum'      # Cumulative returns
})

# Quarter-over-quarter growth
quarterly = stock_data.resample('Q')['price'].last()
quarterly_growth = quarterly.pct_change()

# Example: Time-based features for ML
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
df['is_month_end'] = df.index.is_month_end.astype(int)

# Cyclical encoding (for neural networks)
df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
```

---

## String and Categorical Data

### String Methods (`.str` Accessor)

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'email': ['alice@example.com', 'bob@test.com', 'charlie@example.com', 'david@test.com']
})

# Case conversion
df['upper'] = df['name'].str.upper()
df['lower'] = df['name'].str.lower()
df['title'] = df['name'].str.title()
df['capitalize'] = df['name'].str.capitalize()

# String operations
df['length'] = df['name'].str.len()
df['starts_with_a'] = df['name'].str.startswith('A')
df['ends_with_e'] = df['name'].str.endswith('e')
df['contains_li'] = df['name'].str.contains('li')

# Extract substring
df['first_3'] = df['name'].str[:3]
df['domain'] = df['email'].str.split('@').str[1]

# Replace
df['name_clean'] = df['name'].str.replace('Charlie', 'Chuck')
df['email_clean'] = df['email'].str.replace(r'@.*\.com', '@company.com', regex=True)

# Strip whitespace
df['name_stripped'] = df['name'].str.strip()
df['name_lstrip'] = df['name'].str.lstrip()
df['name_rstrip'] = df['name'].str.rstrip()

# Split
df['email_parts'] = df['email'].str.split('@')
df[['username', 'domain']] = df['email'].str.split('@', expand=True)

# Join
df['full'] = df['name'].str.cat(df['email'], sep=' - ')

# Pad strings
df['name_padded'] = df['name'].str.pad(width=10, side='right', fillchar='*')
df['name_centered'] = df['name'].str.center(10, fillchar=' ')

# Count occurrences
df['count_e'] = df['name'].str.count('e')

# Find position
df['pos_a'] = df['name'].str.find('a')

# Extract using regex
df['username'] = df['email'].str.extract(r'(.*)@', expand=False)

# Extract all matches
df['all_vowels'] = df['name'].str.findall(r'[aeiou]', flags=re.IGNORECASE)

# Boolean checks
df['is_alpha'] = df['name'].str.isalpha()
df['is_digit'] = df['name'].str.isdigit()
df['is_alnum'] = df['name'].str.isalnum()
```

### Categorical Data

Categorical data is useful for columns with a fixed set of values (gender, country, product category, etc.).

```python
# Create categorical
df = pd.DataFrame({
    'grade': ['A', 'B', 'C', 'A', 'B', 'C', 'A']
})

df['grade'] = df['grade'].astype('category')
print(df['grade'].dtype)  # category

# Create ordered categorical
df['grade'] = pd.Categorical(df['grade'], 
                              categories=['C', 'B', 'A'], 
                              ordered=True)

# Access categories
print(df['grade'].cat.categories)

# Access codes (integer representation)
print(df['grade'].cat.codes)

# Add categories
df['grade'] = df['grade'].cat.add_categories(['D', 'F'])

# Remove categories
df['grade'] = df['grade'].cat.remove_categories(['D', 'F'])

# Remove unused categories
df['grade'] = df['grade'].cat.remove_unused_categories()

# Rename categories
df['grade'] = df['grade'].cat.rename_categories({'A': 'Excellent', 'B': 'Good', 'C': 'Average'})

# Reorder categories
df['grade'] = df['grade'].cat.reorder_categories(['Excellent', 'Good', 'Average'])

# Set categories
df['grade'] = df['grade'].cat.set_categories(['A', 'B', 'C', 'D', 'F'])

# Comparisons (only work with ordered categories)
df_filtered = df[df['grade'] > 'B']  # Only if ordered=True

# Memory savings
print(f"Object dtype: {df['grade'].astype('object').memory_usage(deep=True)} bytes")
print(f"Category dtype: {df['grade'].memory_usage(deep=True)} bytes")

# Practical example: Encoding for ML
df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue']
})

# Convert to category
df['color'] = df['color'].astype('category')

# Get numeric codes for ML
df['color_code'] = df['color'].cat.codes

# Or use get_dummies for one-hot encoding
dummies = pd.get_dummies(df['color'], prefix='color')
df = pd.concat([df, dummies], axis=1)
```

---

## Performance Optimization

### Vectorization

```python
import time

# BAD: Python loop
df = pd.DataFrame({'A': range(1000000)})

start = time.time()
result = []
for val in df['A']:
    result.append(val * 2)
df['B'] = result
loop_time = time.time() - start

# GOOD: Vectorized operation
start = time.time()
df['B'] = df['A'] * 2
vectorized_time = time.time() - start

print(f"Loop: {loop_time:.4f}s")
print(f"Vectorized: {vectorized_time:.4f}s")
print(f"Speedup: {loop_time/vectorized_time:.1f}x")

# Use vectorized string methods instead of apply
# BAD
df['upper'] = df['name'].apply(lambda x: x.upper())

# GOOD
df['upper'] = df['name'].str.upper()

# Use NumPy functions when possible
# GOOD
df['log_value'] = np.log(df['value'])

# Use built-in aggregations
# GOOD
mean_value = df['value'].mean()  # Much faster than apply
```

### Method Chaining

```python
# Chain operations for cleaner, more efficient code
result = (df
    .query('age > 25')
    .dropna(subset=['salary'])
    .assign(
        salary_k=lambda x: x['salary'] / 1000,
        age_group=lambda x: pd.cut(x['age'], bins=[0, 30, 50, 100])
    )
    .groupby('age_group')['salary_k']
    .mean()
    .reset_index()
    .sort_values('salary_k', ascending=False)
)

# Use assign() instead of multiple column assignments
# GOOD
df = df.assign(
    col1=df['A'] * 2,
    col2=df['B'] + 10,
    col3=lambda x: x['col1'] + x['col2']  # Can reference newly created columns
)
```

### Memory Optimization

```python
# 1. Use appropriate dtypes
def optimize_dtypes(df):
    """Optimize DataFrame dtypes to reduce memory"""
    for col in df.select_dtypes(include=['int']):
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    for col in df.select_dtypes(include=['object']):
        num_unique = df[col].nunique()
        num_total = len(df)
        if num_unique / num_total < 0.5:  # If less than 50% unique, use category
            df[col] = df[col].astype('category')
    
    return df

# 2. Read only needed columns
df = pd.read_csv('data.csv', usecols=['col1', 'col2', 'col3'])

# 3. Specify dtypes when reading
df = pd.read_csv('data.csv', dtype={
    'id': 'int32',
    'category': 'category',
    'value': 'float32'
})

# 4. Use chunking for large files
chunks = pd.read_csv('large_file.csv', chunksize=50000)
results = []
for chunk in chunks:
    # Process chunk
    processed = chunk[chunk['value'] > 0]
    results.append(processed)
df = pd.concat(results, ignore_index=True)

# 5. Delete intermediate DataFrames
intermediate_df = df.copy()
# ... process ...
del intermediate_df  # Free memory

# 6. Use inplace operations when appropriate
df.drop('col', axis=1, inplace=True)  # Doesn't create copy
df.fillna(0, inplace=True)
```

### Query Optimization

```python
# Use query() for complex boolean indexing (faster and more readable)
# SLOW
result = df[(df['age'] > 25) & (df['salary'] > 50000) & (df['city'] == 'NY')]

# FAST
result = df.query('age > 25 and salary > 50000 and city == "NY"')

# Use eval() for arithmetic expressions
# SLOW
df['new_col'] = df['A'] + df['B'] * df['C']

# FAST
df.eval('new_col = A + B * C', inplace=True)
```

### Avoiding Common Performance Pitfalls

```python
# 1. DON'T grow DataFrames iteratively
# BAD
df = pd.DataFrame()
for i in range(1000):
    df = pd.concat([df, pd.DataFrame({'A': [i]})], ignore_index=True)

# GOOD
data = []
for i in range(1000):
    data.append({'A': i})
df = pd.DataFrame(data)

# 2. DON'T use apply when vectorized operations exist
# BAD
df['result'] = df['A'].apply(lambda x: x * 2 + 1)

# GOOD
df['result'] = df['A'] * 2 + 1

# 3. DON'T use iterrows()
# BAD
for index, row in df.iterrows():
    df.at[index, 'new_col'] = row['A'] * row['B']

# GOOD
df['new_col'] = df['A'] * df['B']

# If you MUST iterate, use itertuples() (faster)
# BETTER (but still slower than vectorization)
for row in df.itertuples():
    # Access as row.A, row.B, etc.
    pass

# 4. Use NumPy when appropriate
# GOOD
df['result'] = np.where(df['A'] > 0, df['B'], df['C'])

# 5. Cache intermediate results
# GOOD
grouped_mean = df.groupby('category')['value'].mean()  # Compute once
df['category_mean'] = df['category'].map(grouped_mean)  # Reuse

# 6. Use appropriate file formats
# CSV: Slow, large files
# Parquet: Fast, compressed, preserves dtypes
df.to_parquet('data.parquet')  # Much faster than CSV
df = pd.read_parquet('data.parquet')
```

---

## Best Practices and Common Pitfalls

### Best Practices

#### 1. Always Specify Index Parameters

```python
# ✅ GOOD: Explicit and clear
df.drop(columns=['col1'])
df.drop(index=[0, 1, 2])

# ❌ AVOID: Ambiguous
df.drop('col1', axis=1)  # Less clear
```

#### 2. Use Method Chaining

```python
# ✅ GOOD: Readable pipeline
result = (df
    .query('age > 25')
    .dropna(subset=['salary'])
    .assign(salary_k=lambda x: x['salary'] / 1000)
    .groupby('department')['salary_k']
    .mean()
    .reset_index()
    .sort_values('salary_k', ascending=False)
)

# ❌ AVOID: Multiple temporary variables
temp1 = df[df['age'] > 25]
temp2 = temp1.dropna(subset=['salary'])
temp3 = temp2.copy()
temp3['salary_k'] = temp3['salary'] / 1000
# ... and so on
```

#### 3. Vectorize Operations

```python
# ✅ GOOD: Vectorized
df['result'] = df['A'] * 2 + df['B']

# ❌ AVOID: Row iteration
for idx, row in df.iterrows():
    df.at[idx, 'result'] = row['A'] * 2 + row['B']
```

#### 4. Use Appropriate Data Types

```python
# ✅ GOOD: Optimize dtypes
df = pd.read_csv('data.csv', dtype={
    'id': 'int32',
    'category': 'category',
    'price': 'float32'
})

# Check memory usage
print(df.memory_usage(deep=True))

# ❌ AVOID: Default dtypes for large datasets
df = pd.read_csv('data.csv')  # May use int64, float64 unnecessarily
```

#### 5. Handle Missing Data Explicitly

```python
# ✅ GOOD: Clear strategy
df = df.dropna(subset=['user_id'])  # Critical column
df['age'] = df['age'].fillna(df['age'].median())  # Reasonable imputation
df['comment'] = df['comment'].fillna('')  # Empty string for text

# ❌ AVOID: Dropping all rows with any NaN
df = df.dropna()  # May lose too much data
```

#### 6. Use Copy When Needed

```python
# ✅ GOOD: Explicit copy to avoid SettingWithCopyWarning
df_subset = df[df['age'] > 25].copy()
df_subset['new_col'] = 0

# ❌ AVOID: Chained assignment
df[df['age'] > 25]['new_col'] = 0  # May not work, throws warning
```

#### 7. Profile Before Optimizing

```python
# ✅ GOOD: Measure before optimizing
import time

start = time.time()
result = expensive_operation(df)
print(f"Time: {time.time() - start:.2f}s")

# Or use line profiler for detailed analysis
%load_ext line_profiler
%lprun -f expensive_operation expensive_operation(df)
```

#### 8. Use Query for Complex Filtering

```python
# ✅ GOOD: Readable and fast
result = df.query('age > 25 and salary > 50000 and department == "IT"')

# ❌ AVOID: Nested boolean indexing
result = df[(df['age'] > 25) & (df['salary'] > 50000) & (df['department'] == 'IT')]
```

### Common Pitfalls

#### 1. SettingWithCopyWarning

```python
# ❌ WRONG: Chained indexing
df[df['A'] > 0]['B'] = 0  # Warning! May not work

# ✅ CORRECT: Use loc
df.loc[df['A'] > 0, 'B'] = 0
```

#### 2. Forgetting `inplace=False` is Default

```python
# ❌ WRONG: Expecting in-place modification
df.dropna()  # Returns new DataFrame, doesn't modify df

# ✅ CORRECT: Either assign or use inplace
df = df.dropna()  # Assign back
# OR
df.dropna(inplace=True)  # Modify in place
```

#### 3. Boolean Operators vs Logical Operators

```python
# ❌ WRONG: Using 'and', 'or', 'not'
df[(df['A'] > 0) and (df['B'] < 10)]  # Error!

# ✅ CORRECT: Use &, |, ~
df[(df['A'] > 0) & (df['B'] < 10)]
```

#### 4. Iteration Instead of Vectorization

```python
# ❌ WRONG: Iterating over DataFrame
for idx in df.index:
    df.loc[idx, 'new_col'] = df.loc[idx, 'A'] * 2

# ✅ CORRECT: Vectorized operation
df['new_col'] = df['A'] * 2
```

#### 5. Not Handling Index After Operations

```python
# ❌ WRONG: Duplicated index after concat
df_combined = pd.concat([df1, df2])  # Index: 0,1,2,0,1,2

# ✅ CORRECT: Reset index
df_combined = pd.concat([df1, df2], ignore_index=True)  # Index: 0,1,2,3,4,5
```

#### 6. Comparing DataFrames with `==`

```python
# ❌ WRONG: Element-wise comparison
if df1 == df2:  # Returns DataFrame, not boolean
    pass

# ✅ CORRECT: Use equals() or array_equal
if df1.equals(df2):
    pass

# Or for element-wise comparison
if (df1 == df2).all().all():
    pass
```

#### 7. Modifying During Iteration

```python
# ❌ WRONG: Modifying DataFrame while iterating
for idx, row in df.iterrows():
    if row['value'] > 10:
        df.drop(idx, inplace=True)  # Dangerous!

# ✅ CORRECT: Boolean indexing
df = df[df['value'] <= 10]
```

#### 8. Not Checking Data Types After Reading

```python
# ❌ WRONG: Assuming correct types
df = pd.read_csv('data.csv')
df['date'].dt.year  # May fail if not datetime

# ✅ CORRECT: Check and convert
print(df.dtypes)
df['date'] = pd.to_datetime(df['date'])
df['category'] = df['category'].astype('category')
```

#### 9. Using `append()` in Loops

```python
# ❌ WRONG: Append in loop (slow, deprecated)
df = pd.DataFrame()
for i in range(1000):
    df = df.append({'A': i}, ignore_index=True)

# ✅ CORRECT: Build list, then create DataFrame
data = []
for i in range(1000):
    data.append({'A': i})
df = pd.DataFrame(data)
```

#### 10. Ignoring Memory Usage

```python
# ❌ WRONG: Not checking memory for large datasets
df = pd.read_csv('huge_file.csv')

# ✅ CORRECT: Monitor and optimize
print(f"Memory: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

# Use chunking or optimize dtypes
df = pd.read_csv('huge_file.csv', dtype={'id': 'int32'})
# Or read in chunks
chunks = pd.read_csv('huge_file.csv', chunksize=50000)
```

### Performance Checklist

✅ Use vectorized operations instead of loops  
✅ Specify dtypes when reading data  
✅ Use categorical for low-cardinality strings  
✅ Avoid chained indexing  
✅ Use `query()` and `eval()` for complex operations  
✅ Read only necessary columns with `usecols`  
✅ Use appropriate file formats (Parquet > CSV)  
✅ Profile code before optimizing  
✅ Use `inplace=True` judiciously (doesn't always save memory)  
✅ Delete intermediate DataFrames with `del`  

### Code Review Checklist

✅ No `SettingWithCopyWarning`  
✅ Explicit `axis` or `index`/`columns` parameters  
✅ Proper handling of missing data  
✅ No unnecessary copies  
✅ Vectorized operations where possible  
✅ Appropriate data types  
✅ Index reset after concatenation if needed  
✅ Proper use of `&`, `|`, `~` in boolean indexing  
✅ `.copy()` when creating subsets for modification  
✅ Memory-efficient operations for large datasets  

---

## References

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [Pandas Cookbook](https://pandas.pydata.org/docs/user_guide/cookbook.html)
- [Category Data Type](https://pandas.pydata.org/docs/user_guide/categorical.html)
- [Nullable Integer Data Type](https://pandas.pydata.org/docs/user_guide/integer_na.html)

---

<h1 align="center"> END </h1>
