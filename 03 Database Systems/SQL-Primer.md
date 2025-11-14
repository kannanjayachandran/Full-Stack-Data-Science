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
    department VARCHAR(40),
    email VARCHAR(100) UNIQUE,
    hire_date DATE,
    salary INTEGER
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
INSERT INTO employees (first_name, last_name, department, email, hire_date, salary)
VALUES 
('Arjun', 'Menon', 'IT-Support', 'arjun_menon@example.com', '2021-03-15', 1200000),
('Priya', 'Nair', 'Development', 'priya_nair@example.com', '2020-11-02', 1500000),
('Clark', 'Dave', 'HR', 'clarky_23@example.com', '2019-08-21', 1100000),
('Sonia', 'Sharma', 'Data Science', 'sonia.sharma@example.com', '2022-01-10', 1800000),
('Rajesh', 'Kumar', 'Development', 'rajesh.kumar@example.com', '2021-06-25', 1650000),
('Anjali', 'Verma', 'Marketing', 'anjali.verma@example.com', '2023-04-01', 950000),
('Vikram', 'Singh', 'Finance', 'vikram.singh@example.com', '2018-12-19', 1400000),
('Deepa', 'Reddy', 'Data Science', 'deepa.reddy@example.com', '2022-09-07', 1750000),
('Mohan', 'Pillai', 'IT-Support', 'mohan.pillai@example.com', '2023-01-30', 1050000),
('Neha', 'Jain', 'Sales', 'neha.jain@example.com', '2021-07-14', 1300000),
('Kiran', 'Rao', 'Development', 'kiran.rao@example.com', '2019-05-11', 1900000),
('Gaurav', 'Bhatia', 'HR', 'gaurav.bhatia@example.com', '2022-10-28', 1150000),
('Shreya', 'Tiwari', 'Marketing', 'shreya.tiwari@example.com', '2023-03-05', 980000),
('Manish', 'Gupta', 'Finance', 'manish.gupta@example.com', '2020-04-17', 1450000),
('Preeti', 'Sinha', 'Data Science', 'preeti.sinha@example.com', '2023-05-20', 1600000);
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

4. **Retrieving unique values using `DISTINCT`**

```sql
SELECT DISTINCT first_name
FROM employees;

-- find all distinct hire years
SELECT DISTINCT EXTRACT(YEAR FROM hire_date) AS hire_year
FROM employees
ORDER BY hire_year ASC;
```

5. **Filtering with `WHERE`**

```sql
--- Returns every single column from the employees table for every row where first_name is exactly 'John'
SELECT *
FROM employees
WHERE first_name = 'John';

-- For particular columns, use this
SELECT hire_date, first_name, last_name
FROM employees
WHERE first_name = 'John';

-- Range check
SELECT *
FROM employees
WHERE hire_date BETWEEN '2020-01-01' AND '2021-03-15'
ORDER BY hire_date ASC

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
SELECT first_name, hire_date
FROM employees
WHERE id IN (1, 2, 3, 4, 5);
```

7. **`LIKE` Operator for pattern matching**

|Wildcard | Description | Example | Matches |
|---|---|---|---|
|`%` (Percent) | Matches any sequence of zero or more characters. | 'A%' | "Anything starting with 'A' (e.g., 'Apple', 'Alley')."|
| `_` (Underscore) | Matches any single character. | '_a%' | "Anything with 'a' as the second character (e.g., 'cat', 'map')."|

```sql
-- Every employee whose first name starts with 'A'
-- LOWER() function is used to get case insensitive results
SELECT first_name
FROM employees
WHERE LOWER(first_name) LIKE LOWER('a%');
```

Postgres has `ILIKE`; which is case insensitive `LIKE`. It is Concise and often highly optimized in PostgreSQL, but not standard.

```sql
SELECT first_name
FROM employees
WHERE first_name ILIKE 'A%';

--- Get all first_name's with the letter 'a' in it
SELECT first_name
FROM employees
WHERE LOWER(first_name) LIKE LOWER('%A%');

--- Get all first_name's that starts and ends with 'a' and has exactly 3 characters in between
SELECT first_name
FROM employees
WHERE LOWER(first_name) LIKE LOWER('A___A');
```

### Aggregate Functions

1. **`COUNT` Function**

```sql
SELECT COUNT(first_name)
FROM employees;
```

2. **`SUM` Function**

```sql
SELECT SUM(salary)
FROM employees;
```

3. **`MIN` Function**

```sql
SELECT MIN(salary)
FROM employees;
```

4. **`MAX` Function**

```sql
SELECT MAX(salary)
FROM employees;
```

5. **`AVG` Function**

```sql
SELECT AVG(salary)
FROM employees;
```

### Grouping and Filtering (`GROUP BY`, `HAVING`)

The main purpose of `GROUP BY` is to aggregate data based on one or more column values, allowing you to perform calculations on each group using aggregate functions.

```sql
-- Avg salary for employees hired in same calendar year along with number of hire using GROUP BY
SELECT EXTRACT(YEAR FROM hire_date) AS hire_year, 
AVG(salary) AS avg_salary, 
COUNT(id) AS no_hire
FROM employees
GROUP BY hire_year
ORDER BY hire_year ASC;
```

```sql
-- All hire years where the average salary of that year was greater than 1300000
SELECT EXTRACT(YEAR FROM hire_date) AS hire_year,
AVG(salary) AS avg_salary
FROM employees
GROUP BY hire_year
HAVING AVG(salary) > 1300000
ORDER BY hire_year;
```

```sql
-- Department wise avg salary
SELECT department, AVG(salary) AS avg_salary
FROM employees
GROUP BY department
ORDER BY avg_salary ASC;
```

```sql
-- Find the oldest hire date for each unique salary level
SELECT salary, MIN(hire_date) as min_hire_date
FROM employees
GROUP BY salary
ORDER BY salary DESC;
```

- `WHERE` filters individual rows before grouping.
- `HAVING` filters groups after grouping.

## Joins in SQL

```sql
CREATE TABLE projects (
    project_id SERIAL PRIMARY KEY,
    project_name VARCHAR(100) NOT NULL,
    start_date DATE NOT NULL,
    status VARCHAR(20)
);
```

```sql
CREATE TABLE employee_projects (
    employee_id INTEGER REFERENCES employees(id),  -- Foreign Key to employees
    project_id INTEGER REFERENCES projects(project_id), -- Foreign Key to projects
    hours_worked INTEGER,
    role VARCHAR(50),
    PRIMARY KEY (employee_id, project_id) -- Composite Primary Key
);
```

```sql
INSERT INTO projects (project_name, start_date, status)
VALUES 
('ML Recommendation Engine', '2023-08-01', 'In Progress'),
('HR Data Analytics Dashboard', '2023-05-15', 'Completed'),
('Cloud Migration Phase 2', '2024-01-10', 'In Progress'),
('Mobile App Relaunch', '2022-11-20', 'Testing');
```

```sql
INSERT INTO employee_projects (employee_id, project_id, hours_worked, role)
VALUES 
(4, 1, 350, 'Lead Data Scientist'),    
(8, 1, 280, 'ML Engineer'),            
(2, 3, 400, 'Senior Developer'),       
(5, 3, 320, 'Developer'),              
(1, 4, 150, 'Technical Support'),      
(3, 2, 80, 'HR Analyst'),              
(12, 2, 50, 'HR Specialist'),          
(10, 4, 250, 'Sales Strategy Lead'),   
(11, 3, 500, 'Project Lead'),          
(15, 1, 190, 'Data Analyst');
```
### Inner JOIN

Match in both tables

```sql
SELECT 
    e.first_name, 
    e.last_name, 
    p.project_name
FROM 
    employees e
JOIN 
    employee_projects ep ON e.id = ep.employee_id
JOIN 
    projects p ON ep.project_id = p.project_id
WHERE 
    p.project_name = 'ML Recommendation Engine';
```
