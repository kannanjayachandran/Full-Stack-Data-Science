<!-- 
    Author : Kannan Jayachandran
    File : Linear_Algebra.md
    Section : Mathematics for data science and machine learning
 -->

<h1 align="center"> Linear Algebra </h1>

## Table of contents

1. [Linear Equations](#Linear-equations)
1. [Systems of linear equations](#Systems-of-linear-equations)

---

**Linear algebra is the branch of mathematics that deals with linear equations and their representations in vector spaces, through matrices.**

<!-- SECTION - I -->

## Linear equations

A **linear equation** is an equation that can be written in the form: 

$$a_1x_1 + a_2x_2 + ... + a_nx_n + b = 0$$

Where:

- $x_1, x_2, ..., x_n$ are the **variables** (Unknowns),

- $a_1, a_2,  ..., a_n$ are the **coefficients**,

- $b$ is a **constant** term.

> Example of a linear equation is $2x_1 + 3x_2 - 4 = 0$.

## Systems of linear equations

A **system of linear equations** is a set of linear equations that involve the same set of variables. In general, a system with $𝑛$ variables and $m$ equations can be written as:

$$a_{11}x_1 + a_{12}x_2 + a_{13}x_3 + ... + a_{1n}x_n = b_1$$

$$a_{21}x_1 + a_{22}x_2 + a_{23}x_3 + ... + a_{2n}x_n = b_2$$

$$a_{31}x_1 + a_{32}x_2 + a_{33}x_3 + ... + a_{3n}x_n = b_3$$

$$...$$

$$a_{m1}x_1 + a_{m2}x_2 + a_{m3}x_3 + ... + a_{mn}x_n = b_m$$

where:

- $x_1, x_2, x_3, ..., x_n$ are the variables,

- $a_{ij}$ are the coefficients of the variables,

- $b_i$ are the constant terms.

### Solutions of a System of Linear Equations

A system of linear equations can have:

1. **No solution** (the system is inconsistent and singular),

2. **Exactly one solution** (the system is non-singular, consistent and independent), or

3. **Infinitely many solutions** (the system is consistent and dependent).

To visualize this, let us plot two linear equations in two variables on a graph:

![Graph of equations ](./img/graph_solution.png)

Thus, two fundamental questions about a system of linear equations are:

1. Is the system consistent? (i.e., does at least one solution exist?)

2. If the system is consistent, is the solution unique? (i.e., is there exactly one solution?)

### Methods to Solve a System of Linear Equations

There are several methods to solve a system of linear equations, such as:

- Substitution method

- Elimination method

- Matrix inversion

- Gaussian elimination

- Gauss-Jordan elimination

The above system of equations can be written in matrix form as:

$$Ax = b$$

Where:

- $A$ is an $m X n$ matrix of coefficients, which looks like:

$$A = \begin{bmatrix} a_{11} & a_{12}  & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ a_{31} & a_{32} & \cdots & a_{3n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}$$

- $x$ is the vector of variables,

$$x = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}$$

- $b$ is the vector of constants.

$$b = \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_m \end{bmatrix}$$ 

<!-- SECTION - II -->

## Vector

A **vector** is a collection or list of numbers, often used to describe the state or properties of a system in mathematics, physics, and engineering. Vectors can represent anything from forces to velocities in different dimensions.

![Point or vector on cartesian coordinate system](./img/Point_vector.png)
>Point or vector on cartesian coordinate system

An **n-dimensional** vector can be represented as $[x_1, x_2, x_3, ..., x_n]$

Where:

- $n$ is the length of the vector

- $x_i$ is a component of the vector in the **i-th** dimension.

> For example, a 3-dimensional vector could be written as: $[3, -4, 2]$

## Row and Column Vectors

Row vectors are vectors that consist of a single row and multiple columns. For example:

$$[1, 2, 3]$$

Column vectors are vectors that consist of a single column and multiple rows. For example:

$$\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$$

Taking the transpose of a row vector converts it into a column vector, and vice versa.

## Vector Dot Product

The **dot product** (also known as the scalar product) of two vectors is a scalar quantity, which is the sum of the products of their corresponding components. It is defined only for vectors of the same length.

The dot product of two vectors $a$ and $b$ is denoted by  $a \cdot b$, and is computed as:

$$a \cdot b = a_1b_1 + a_2b_2 + ... + a_nb_n$$

Alternatively, this can be expressed as the product of a row vector and a column vector:

$$[a_1, a_2, a_3, ..., a_n] \; \cdot \; \begin{bmatrix} b_1 \\ b_2 \\ b_3 \\ ... \\ b_n \end{bmatrix}$$

![Matrix representation of linear equation](./img/matrix_form_linear_equation.png)

### Geometrically interpretation of dot product.

![Geometrical interpretation of dot product](./img/dot_product.png)


In an **n-dimensional space**, the dot product of two vectors $a$ and $b$ can be expressed as:

$$a.b = \sum_{i=1}^{n}a_ib_i = |a||b| \; cos\theta$$

where:

- $a = [a_1, a_2, a_3, ..., a_n]$ and $b = [b_1, b_2, b_3, ..., b_n]$

- $|a|$ and $|b|$ are the magnitudes(or lengths) of the vectors $a$ and $b$ ($|a| = \sqrt{a_1^2 + a_2^2 + a_3^2 + ... + a_n^2}$)

- $\theta$ is the angle between the two vectors.

> If the two vectors are perpendicular (i.e., $\theta = 90^{\circ}$), then $\cos \theta = 0$ and the dot product is zero. 

> If the two vectors are parallel (i.e., $\theta = 0^{\circ}$ or $180^\circ$), then the dot product is equal to the product of their magnitudes.

## Inner product

The **inner product** is a generalization of the dot product that operates in any vector space, not just in Euclidean spaces. It is a function that takes in two vectors and returns a scalar, providing information about the "`correlation`" or "`similarity`" between the vectors. The inner product is typically denoted by:

$$\langle x, y \rangle$$

For two vectors $x$ and $y$, the inner product is defined as:

$$\langle x, y \rangle = x^Ty = \sum_{i=1}^{N}x_iy_i$$

where:

- $x = [x_1, x_2, x_3, ..., x_n]$,

- $y = [y_1, y_2, y_3, ..., y_n]$,

- $N$ is the number of dimensions (or length) of the vectors.

If the vectors are **correlated** or nearly parallel, the inner product will be large. f the vectors are close to **perpendicular**, the inner product will be small or zero.

> Inner product is used by Hilbert space (Hilbert spaces allow the methods of linear algebra and calculus to be generalized from Euclidean vector spaces to spaces that may be infinite-dimensional.)

![Inner product image](./img/Inner_product.png)
>Geometric interpretation of inner product

The inner product $\langle x, y \rangle$ can be geometrically interpreted as the length of the projection of vector $y$ onto vector $x$ multiplied by the length of vector $x$.

## Vector norm

The **norm** of a vector is a measure of its length or magnitude. It quantifies the size or extent of the vector in its space. The norm of a vector $a$ is denoted by:

$$||a||$$

For an **n-dimensional** vector $a = [a_1, a_2, ..., a_n]$, the most commonly used norm (the Euclidean norm) is given by:

$$||a|| = \sqrt{a_1^2 + a_2^2 + a_3^2 + ... + a_n^2}$$

where $\color{#F99417}a = [a_1, a_2, a_3, ..., a_n]$ is a vector.

> Vector norm is always **non-negative**, and it describes the "length" or "distance" of the vector from the origin in the vector space.

### $L^1$ Norm

$L^1$ Norm or also called the **Manhattan** norm, is the sum of the absolute values of a vector’s components. It is denoted by $||a||_1$ and is defined as:

$$||a||_1 = |a_1| + |a_2| + |a_3| + ... + |a_n|$$

where:

- $a = [a_1, a_2, ..., a_n]$ is the vector.

> It is often used in machine learning as a regularization technique (known as **Lasso regression**) to keep model coefficients small and avoid overfitting.

### $L^2$ Norm

$L^2$ Norm also known as **Euclidean** norm, is the square root of the sum of the squares of the vector components. It is denoted by $||a||_2$ and is given by;

$$||a||_2 = \sqrt{a_1^2 + a_2^2 + a_3^2 + ... + a_n^2}$$

where $\color{#F99417}a = [a_1, a_2, a_3, ..., a_n]$ is a vector.

> It is used as a regularization technique in machine learning (known as **Ridge regression**) to prevent overfitting.

### $L^\infin$ Norm

**Vector max norm** or **maximum norm** or **supremum norm**  also called $L^\infin$ Norm is the largest absolute value of the components of the vector. It is denoted by $||a||_\infin$ and is given by;

$$||a||_\infin = max(|a_1|, |a_2|, |a_3|, ..., |a_n|)$$

> It useful in optimization problems where we are concerned with the maximum deviation of a vector’s components.

![3 vector norms graphical equivalent](./img/3_norms.png)

## Projection of a vector

The projection of a vector $a$ onto another vector $b$, denoted as $proj_b(a)$, is a vector $p$ that lies in the direction of $b$ and is collinear with $a$. This projection represents the component of $a$ that points along the direction of $b$.

![Projection of a vector](./img/Projection.png)

Mathematically the projection of $a$ onto $b$ is given by: 

$$proj_b(a) = \frac{a \cdot b}{||b||^2}b$$

where:

- $a \cdot b$ is the dot product of vectors $a$ and $b$,

- $||b||^2$ is the square of the norm (or length) of vector $b$,

- $b$ is the vector in the direction of the projection.

We can write this also as:


$$d = \frac{a.b}{||b||} = d = ||a||cos\theta$$

where:

- $d$ is the scalar projection of $a$ onto $b$,

- $\theta$ is angle between the two vectors.

> If $a$ and $b$ are orthogonal (perpendicular), their dot product is zero, meaning the projection of $a$ onto $b$ is a zero vector.

## Unit vector

**Unit vector** is a vector with a magnitude (or norm) of 1. It is typically used to indicate direction without regard to magnitude. The unit vector of any vector $a$ is denoted by $\hat{a}$, points in the same direction as $a$, and is calculated as:

$$\hat{a} = \frac{a}{||a||}$$

where:

- $||a||$ is the magnitude of the original vector $a$.

> Unit vector preserves the direction but normalizes the magnitude of the vector to a length of 1.

<!-- EDITED TILL HERE -->

<!-- SECTION - III -->

## Matrix 

A matrix is a 2d array of numbers, with one or more rows and one or more columns. A matrix with `m` rows and `n` columns is said to have the dimension `m x n`. While multiplying matrices we need to ensure that;

$$\color{#FF9900}\text{No. of Col(}m_1\text{) = No. of rows(}m_2)$$

## Types of matrices

- **Square matrix** : A matrix with the same number of rows and columns. For example; $\color{#F99417}X = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}$

- **Diagonal matrix** : A square matrix with all the elements outside the main diagonal are zero. For example; $\color{#F99417}X = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 5 & 0 \\ 0 & 0 & 9 \end{bmatrix}$

- **Identity matrix** : A diagonal matrix with all the elements in the main diagonal are one. For example; $\color{#F99417}X = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$

- **Upper triangular matrix** : A square matrix with all the elements below the main diagonal are zero. For example; $\color{#F99417}X = \begin{bmatrix} 1 & 2 & 3 \\ 0 & 5 & 6 \\ 0 & 0 & 9 \end{bmatrix}$

- **Lower triangular matrix** : A square matrix with all the elements above the main diagonal are zero. For example; $\color{#F99417}X = \begin{bmatrix} 1 & 0 & 0 \\ 4 & 5 & 0 \\ 7 & 8 & 9 \end{bmatrix}$

- **Symmetric matrix** : A square matrix that is equal to its transpose. For example; $\color{#F99417}X = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 5 & 6 \\ 3 & 6 & 9 \end{bmatrix}$

- **Skew-symmetric matrix** : A square matrix that is equal to the negative of its transpose. For example; $\color{#F99417}X = \begin{bmatrix} 0 & 2 & 3 \\ -2 & 0 & 6 \\ -3 & -6 & 0 \end{bmatrix}$

- **Orthogonal matrix** : A square matrix whose rows and columns are orthonormal unit vectors. For example; $\color{#F99417}X = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$

    - It is defined as $\color{#F99417}Q^T\cdot Q = I$ where $I$ is the identity matrix. Therefore we can say that a matrix is orthogonal if its transpose is equal to its inverse $\color{#F99417}Q^T = Q^{-1}$.

    > Two vectors are `unit vectors` if their magnitude is one. Two vectors are `orthogonal` if their dot product is zero. If two vectors are orthogonal and unit vectors, then they are called `orthonormal vectors`.

- **Sparse matrix** : A matrix with a large number of zero elements. For example; $\color{#F99417}X = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 9 \end{bmatrix}$

On the other hand a matrix with a large number of non-zero elements is called a `dense matrix`. Sparsity score of a matrix is the ratio of the number of zero elements to the total number of elements in the matrix.

$$\text{Sparsity score} = \frac{\text{No. of zero elements}}{\text{Total no. of elements}}$$

## Transpose of a matrix

The transpose of a matrix is a new matrix whose rows are the columns of the original. It is denoted by $\color{#F99417}X^T$ and is given by;

$$X = \begin{bmatrix} a & b & c \\ d & e & f \end{bmatrix} \implies X^T = \begin{bmatrix} a & d \\ b & e \\ c & f \end{bmatrix}$$

## Inverse of a matrix

The inverse of a matrix is a matrix that when multiplied with the original matrix gives the identity matrix. It is denoted by $\color{#F99417}X^{-1}$ and is given by;

$$X = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \implies X^{-1} = \frac{1}{ad-bc}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

where $\color{#F99417}ad-bc$ is called the `determinant` of the matrix.

> The inverse of a matrix does not exist if the determinant of the matrix is zero.

- A square matrix is invertible if and only if its determinant is non-zero. An invertible square matrix is called a `non-singular matrix`. A square matrix that is not invertible is called a `singular matrix`.

## Trace of a matrix

The trace of a matrix is the sum of the elements in the main diagonal of the matrix. It is denoted by $\color{#F99417}tr(X)$ and is given by;

$$X = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \implies tr(X) = a + d$$

## Determinant of a matrix

Determinant of a matrix is a special value which gives us a scalar representation of the matrix volume. It is the product of the eigenvalues of the matrix. It is denoted by $\color{#F99417}det(X) \;or \;|X|$ and is given by;

$$X = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \implies det(X) = ad-bc$$

![Formula for finding determinant](./img/Determinant_formula.png)

### The intuition behind determinant

The determinant of a matrix is a scalar value that tells us how much the matrix scales the volume (or area) spanned by its column vectors (The area spanned by column vectors means the area of the parallelogram formed by the column vectors.). 
- A determinant of 1 indicates that the matrix preserves volume (or area) without scaling it. 

- A determinant greater than 1 indicates that the matrix expands the volume (or area).

- A determinant less than 1 indicates that the matrix shrinks the volume (or area).

- A negative determinant indicates that the matrix flips the orientation (a reflection) in addition to scaling.

- If the determinant of the matrix is zero, it means that the matrix does not have an inverse and the rows or columns of the matrix are linearly dependent. This also means the matrix squishes the space down to a lower dimension. In the context of linear transformations, it signifies that the transformation is not one-to-one, and thus, it cannot be inverted uniquely. Thus there are infinitely many solutions (inverses) or none at all, depending on the specific case.

![Area enclosed by parallelogram](./img/Parallelogram.png)

## Rank of a matrix

The rank of a matrix is the maximum number of linearly **independent** rows or columns in the matrix. It is denoted by $\color{#F99417}rank(X)$. We can find the rank of a matrix by reducing it to its `row echelon form` or `reduced row echelon form` and counting the number of non-zero rows. We use matrix decomposition techniques to find the rank of a matrix.

### Intuition behind rank of a matrix

The rank of a matrix is the number of dimensions spanned by its column vectors. 

- Rank 0 means that all the vectors span a point (zero dimensions).

- Rank 1 means that all the vectors span a line (one dimension).

- Rank 2 means that all the vectors span a plane (two dimensions).

- Rank 3 means that all the vectors span a volume (three dimensions).

## Tensors

A tensor is a generalization of vectors and matrices and is easily understood as a multidimensional array. A tensor can be represented as a `scalar`, `vector`, `matrix`, or `n-dimensional` array. A tensor with `n` dimensions is said to have a `rank` of `n`.

> $\color{#F99417}Scalar → Vector → Matrix → Tensor$

All the arithmetic operation we can do with matrices can be done with tensors. We can add, subtract, multiply (both `Hadamard` and dot product), and divide tensors. 

### Tensor dot product

The dot product of two tensors is a generalization of the dot product of two vectors. It is denoted by $\color{#F99417}a ⊗ b$ and is given by;

$$a \cdot b = \sum_{i=1}^{n}a_ib_i$$

where $\color{#F99417}a$ and $\color{#F99417}b$ are two tensors.

<!-- SECTION - IV -->

## Line

A line is a `one-dimensional` figure that extends infinitely in one directions, having length but no width, depth or curvature. Eg. A ray of light, the number line, etc. 

In a `two dimension`, a line is defined as the collection of points $\color{#FF9900}(x, y)$ that satisfies a linear equation:

$$ax + by + c = 0$$

$$or$$

$$w_{1}x_1 + w_2x_2 + w_0 = 0$$

In `three dimension` instead of a line, we have a **plane**. A plane is a collection of points $\color{#FF9900}x, y, z$ that satisfies the equation:

$$ ax + by + cz + d = 0$$

In `n-dimensional` space, we generalize this further to an `hyperplane`, which is a collection of points $\color{#FF9900}x_1, x_2, ..., x_n$ that satisfies the equation:

$$a_1x_1 + a_2x_2 + a_3x_3 + ... + a_nx_n + b = 0$$

$$or$$

$$w_{1}x_1 + w_2x_2 + w_3x_3 + ... + w_nx_n + w_0 = 0$$

Which can be simplified using vector notation as:

$$ w^Tx + w_0 = 0$$

> This is also the equation for a hyperplane in an $n$ - dimensional space ($\pi_n$).

where $\color{#FF9900}w = [w_1, w_2, w_3, ..., w_n]$  and $\color{#FF9900}x = [x_1, x_2, x_3, ..., x_n]$ are column vectors. If the hyperplane passes through the origin, $w_0 = 0$, then the equation simplifies to:

$$w^Tx=0$$

> 0-Dimension : (A point) -> 1-Dimension : (A line) -> 2-Dimension : (A plane) -> n-dimension : (Hyperplane)

## Distance of a point from a plane

![Distance of a point from a plane](./img/point_plane.png)

Let us consider a plane $\color{#FF9900}\pi_n$ that passes through origin in an `n-dimensional` space and a point $\color{#FF9900}P$ located at the coordinates $\color{#FF9900}x_1, x_2, ..., x_n$. The distance between the point $\color{#FF9900}P$ and the plane $\color{#FF9900}\pi_n$ is given by the formula:

$$d = \frac{w^Tp}{||w||}$$

where:
- $\color{#FF9900}w$ is the normal vector to the plane $\color{#FF9900}\pi_n$,

- $\color{#FF9900}||w||$ is the magnitude of the vector $\color{#FF9900}w$ (or Euclidean norm).

- $\color{#FF9900}w^Tp$ represents the dot product of the normal vector $\color{#FF9900}w$ and the point vector $\color{#FF9900}P$.

The distance represents the perpendicular distance from the point to the plane. Similarly we can compute the distance from the plane to another  point $\color{#FF9900}P^{'}$ as;

$$d^{'} = \frac{w^Tp^{'}}{||w||}$$

In the above diagram, the distance $\color{#FF9900}d$ is positive because the angle between the normal vector $\color{#FF9900}w$ and the point vector $\color{#FF9900}P$ is less than $\color{#FF9900}90^{\circ}$. On the other hand, the distance $\color{#FF9900}d^{'}$ is negative because the angle between the normal vector $w$ and the point vector $\color{#FF9900}P^{'}$ is greater than $\color{#FF9900}90^{\circ}$, meaning that the point $\color{#FF9900}P^{'}$ lies on the opposite side of the plane.

> A positive distance means the point lies on the same side as the normal vector $w$, while a negative distance means it lies on the opposite side.

When calculating the physical distance between a point and a plane, we typically take the absolute value of the signed distance. This ensures that the distance is always non-negative, as negative physical distance does not make sense. However, the sign of the distance is important in determining the relative position of the point with respect to the plane

Some of the common **distance formulas** are;

| Description | Formula |
| --- | :--- |
| Distance between origin $\color{#F99417}o(0, 0)$ and a point $\color{#F99417}P(x_1, x_2)$ in a 2D plane | $d = \sqrt{x_1^2+x_2^2}$ |
| Distance between origin $\color{#F99417}o(0, 0, 0)$ and a point $\color{#F99417}P(x_1, x_2, x_3)$ in a 3D plane | $d = \sqrt{x_1^2+x_2^2+x_3^2}$ |
| Distance between origin $\color{#F99417}o(0, 0, 0, ..., 0)$ and a point $\color{#F99417}p(x_1, x_2, x_3, ..., x_n)$ in a `n-dimensional` plane | $d = \sqrt{x_1^2+x_2^2+x_3^2+...+x_n^2}$ |
| Distance between two points  $\color{#F99417}P (x_1, y_1)$ and $\color{#F99417}Q (x_2, y_2)$ in a 2D plane | $d = \sqrt{(x_2 - x_1)^2+(y_2 - y_1)^2}$ |
| Distance between two points  $\color{#F99417}P (x_1, y_1, z_1)$ and $\color{#F99417}Q (x_2, y_2, z_2)$ in a 3D plane | $d = \sqrt{(x_2 - x_1)^2+(y_2 - y_1)^2+(z_2 - z_1)^2}$ |
| Distance between two points $\color{#F99417} P(x_1, x_2, x_3, ..., x_n)$ and $\color{#F99417}Q (y_1, y_2, y_3, ..., y_n)$ in a `n-dimensional` plane | $d = \sqrt{(x_1-y_1)^2+(x_2 - y_2)^2+(x_3 - y_3)^2+...+(x_n - y_n)^2} \\  \;\;\;\;\;{ or }\\ d = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$ |

> All of these formulas represent the Euclidean distance, providing a way to measure the distance between two points in different-dimensional spaces. We can easily derive all the above formulas using the Pythagoras theorem.

## Slope

The slope of a line is a measure of how steep the line is. It is denoted by $\color{#F99417}m$ and is given by;

$$m = \frac{y_2 - y_1}{x_2 - x_1}$$

where $\color{#F99417}(x_1, y_1)$ and $\color{#F99417}(x_2, y_2)$ are two points on the line.

## Intercept

The intercept of a line is the point where the line crosses the y-axis. It is denoted by $\color{#F99417}c$ and is given by;

$$c = y - mx$$

where $\color{#F99417}(x, y)$ is a point on the line and $\color{#F99417}m$ is the slope of the line.

## Circle

A circle is a collection of points that satisfy the equation $\color{#FF9900} (x)^2 + (y)^2 = r^2$ where $\color{#FF9900}r$ is the radius of the circle and it's center is at the origin $\color{#FF9900}(0, 0)$. The general equation of a circle with center $\color{#FF9900}(h, k)$ and radius $\color{#FF9900}r$ is given by;

$$(x - h)^2 + (y - k)^2 = r^2$$

Given a point $\color{#FF9900}p(x_1, x_2)$, we can determine whether that point lies inside the circle, on the circle, or outside the circle.

- If $\color{#FF9900}x_1^2 + x_2^2 < r^2$, the point lies inside the circle.

- If $\color{#FF9900}x_1^2 + x_2^2 = r^2$, the point lies on the circle.

- If $\color{#FF9900}x_1^2 + x_2^2 > r^2$, the point lies outside the circle.

In `3D`, we have **sphere** instead of circle. The general equation for a circle with center $\color{#FF9900}(h, k, l)$ and radius $r$ is given by;

$$(x_1 - h)^2 + (x_2 - k)^2 + (x_3 - l)^2 = r^2$$

A higher dimensional sphere or a **Hypersphere** is defined as;

$$(x_1 - h)^2 + (x_2 - k)^2 + (x_3 - l)^2 + ... + (x_n - m)^2 = r^2$$

If the center of the hypersphere is at the origin, then the equation of the hypersphere is given by;

$$x_1^2 + x_2^2 + x_3^2 + ... + x_n^2 = r^2 \implies \sum_{i=0}^{n}x_i^2 = r^2$$

The same idea of a point inside a circle or not using the equation of a circle can be extended to higher dimensions. This again is pretty powerful as we can use this idea to determine whether a point lies inside a hyper-sphere or not.

---

## Notations

| Notation | Description |
| --- | :--- |
| $\color{#F99417}a$ | Scalar or Vector |
| $\color{#F99417}A, B, C$ | Matrix |
| $\color{#F99417}A$ of size $\color{#F99417}\text{m X n}$ | Matrix `A` with `m` rows and `n` columns  |
| $\color{#F99417}A_{ij}$ | Element in the `i-th` row and `j-th` column of matrix `A` |
| $\color{#F99417}A^T$ | Transpose of matrix `A` |
| $\color{#F99417}v^T$ | Transpose of vector `v` |
| $\color{#F99417}A^{-1}$ | Inverse of matrix `A` |
| $\color{#F99417}A^*$ | Conjugate transpose of matrix `A` |
| $\color{#F99417}det(A)$ | Determinant of matrix `A` |
| $\color{#F99417}AB$ | Matrix multiplication of matrix `A` and matrix `B`|
| $\color{#F99417}u.v; \langle u, v\rangle$ | Dot product of `u` and `v`|
| $\color{#F99417}u \times v$ | Cross product of `u` and `v`|
| $\color{#F99417}\R$ | Set of real numbers (set $\R$ is infinite and continuous)|
| $\color{#F99417}\R^2$ | Two dimensional real vector space|
| $\color{#F99417}\R^n$ | n-dimensional real vector space |
| $\color{#F99417}v\in\R^n$ | Vector `v` belongs to the space $\R^n$|
| $\color{#F99417}\|v\|_1$ | L1 Norm or Manhattan distance of the vector $v$|
| $\color{#F99417}\|v\|_2; \|\|v\|\|$ | L2 Norm or Euclidean norm of the vector $v$|
| $\color{#F99417}\|v\|_\infin$ | Infinity Norm or Maximum Norm or Chebyshev Norm of vector $v$|
| $\color{#F99417}T: \R^n \rightarrow \R^m;T(v)=w$ | Transformation `T` of a vector `v` $\in \R^n$ into the vector `w` $\in \R^m$|

---

### [Jupyter notebook of linear algebra](./Notebooks/Linear_algebra.ipynb)

**Checkout [Calculus](./Calculus.md)**
