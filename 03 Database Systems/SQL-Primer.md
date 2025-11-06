<h1 align='center'> SQL Primer </h1>

> SQL : Structured Query Language

## Basic Terminology

| Term | Definition | Example |
|------|------------|---------|
| **Data Value** | Single unit of data | `42`, `'Alice'`, `2024-01-15` |
| **Record/Row** | Collection of related values representing one entity | User record: `(1, 'Alice', 'alice@email.com')` |
| **Field/Column** | Attribute of an entity | `user_id`, `name`, `email` |
| **Table/Relation** | Collection of records with same structure | `users`, `orders`, `products` |
| **Schema** | Blueprint defining database structure | Tables, relationships, constraints |
| **Result Set** | Temporary table returned by a query | Output of `SELECT` statement |

## Database Schema

A **database schema** defines the logical structure of the data: tables, fields, constraints, relationships, and indexes.

> The syntax of SQL queries can vary slightly depending on the database engine being used (e.g., MySQL, PostgreSQL, SQLite).

## Query Execution Order

Understanding the **logical** order of SQL query execution is crucial for writing efficient queries:

```sql
-- SQL is written in this order:
SELECT column1, column2
FROM table1
JOIN table2 ON condition
WHERE filter_condition
GROUP BY column1
HAVING aggregate_condition
ORDER BY column1
LIMIT 10;

-- But executed in this order:
-- 1. FROM/JOIN       - Determine source tables and combine them
-- 2. WHERE           - Filter individual rows
-- 3. GROUP BY        - Group rows by specified columns
-- 4. HAVING          - Filter groups
-- 5. SELECT          - Select columns (including aggregations)
-- 6. DISTINCT        - Remove duplicates (if specified)
-- 7. ORDER BY        - Sort results
-- 8. LIMIT/OFFSET    - Restrict number of rows returned
```

- **We are going to use Postgres**.

## Connecting to PostgresSQL

```sh
# Connect as the postgres superuser
psql -U postgres
```

There are two type of commands we can can execute now;

1. **SQL Commands**: Standard SQL commands, they must end with semicolon (`;`)
2. **Postgres Meta Commands**: They are shortcuts for `psql` itself, no need of `;` for them.

## Managing Database

1. **Showing Database**

```sh
\l
```

2. **Create a new Database**

```sql
CREATE DATABASE first_db_psg;
```

3. **Connect to a specific database**

```sh
\c first_db_psg
```

## Managing Users (Roles)

1. **Create a new User**

```sql
CREATE USER app_user WITH PASSWORD 'a very strong password🪿';
```

2. **Grant Permissions**

By default a new user can't do anything. Permissions should be granted explicitly.

```sql
GRANT ALL PRIVILEGES ON DATABASE first_db_psg TO app_user;
```

3. **Change User's password**

```sql
ALTER USER app_user WITH PASSWORD 'a new stronger password 🐼';
```

## Managing Tables

1. **Create Table**

```sql
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(30) NOT NULL,
    last_name VARCHAR(30),
    email VARCHAR(100) UNIQUE,
    hire_date DATE
);
```

2. **Show existing tables**

```sh
\dt
```

3. **Inspect a table's structure**

```sh
\d employees

# detailed schema
\d employees
```

4. **Inserting data into a table**

```sql
INSERT INTO employees (first_name, last_name, email, hire_date)
VALUES 
('Arjun', 'Menon', 'arjun.menon@example.com', '2021-03-15'),
('Priya', 'Nair', 'priya.nair@example.com', '2020-11-02');
```

5. **Inserting data from another table**

```sql
-- Create employee_new with the same structure
CREATE TABLE employee_new (LIKE employees INCLUDING ALL);

-- Copy the first 1000 rows
INSERT INTO employee_new
SELECT *
FROM employees
WHERE id <= 1000;
```

If new table has different schema, we need to explicitly copy columns;

```sql
INSERT INTO employee_new (first_name, last_name, email, hire_date)
SELECT first_name, last_name, email, hire_date
FROM employees
WHERE id <= 1000;
```

6. **Updating data in a table**

```sql
UPDATE employees
SET first_name = 'Rahul'
WHERE id = 1;
```

7. **Deleting data from a table**

```sql
DELETE FROM employees
WHERE id = 2;
```

> **Tip**: Always pair DELETE with a WHERE clause to prevent removing all rows.

## Retrieving Data

1. **Select all columns**

```sql
SELECT * FROM employees;
```

2. **Select specific columns**

```sql
SELECT hire_date FROM employees;
```

3. **Limiting rows**

```sql
SELECT hire_date
FROM employees
LIMIT 100;
```

4. **Ordering using `ORDER BY`**

```sql
SELECT first_name, last_name
FROM employees
ORDER BY first_name ASC;
```

4. **Retrieving unique values using `UNIQUE`**

```sql
SELECT DISTINCT first_name
FROM employees;
```

5. **Filtering with `WHERE`**

```sql
-- Equality
SELECT *
FROM employees
WHERE first_name = 'John';

-- Range check
SELECT *
FROM employees
WHERE hire_date BETWEEN '2020-01-13' AND '2021-03-15';

-- NULL values
SELECT first_name, last_name
FROM employees
WHERE last_name IS NOT NULL;
```

6. **Logical Operators**

```sql
-- AND
SELECT *
FROM employees
WHERE first_name = 'Rahul' AND hire_date = '2021-03-15';

-- OR
SELECT *
FROM employees
WHERE first_name = 'Rahul' OR last_name = 'Doe';

-- NOT
SELECT *
FROM employees
WHERE NOT hire_date = '2021-03-15';

-- IN
SELECT first_name
FROM employees
WHERE id IN (1, 2, 3, 4, 5);
```
