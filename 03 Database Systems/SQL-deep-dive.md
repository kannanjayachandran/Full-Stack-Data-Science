<h1 align='center'> SQL Deep Dive </h1>

## Table of Contents


1. [Advanced SELECT Techniques](#advanced-select-techniques)
1. [Window Functions for Analytics](#window-functions-for-analytics)
1. [Complex Joins and Set Operations](#complex-joins-and-set-operations)
1. [Subqueries vs CTEs](#subqueries-vs-ctes)
1. [Date and Time Operations](#date-and-time-operations)
1. [String Manipulation](#string-manipulation)
1. [Conditional Logic with CASE](#conditional-logic-with-case)
1. [Advanced Aggregations](#advanced-aggregations)
1. [Query Optimization Techniques](#query-optimization-techniques)
1. [Feature Engineering with SQL](#feature-engineering-with-sql)
1. [Time-Series Analysis Patterns](#time-series-analysis-patterns)
1. [Data Quality and Validation](#data-quality-and-validation)
1. [Interview Problems and Solutions](#interview-problems-and-solutions)

---

## Essential `SELECT` Patterns

1. **Column Selection and Aliasing**

```sql
-- Select specific columns with readable aliases
SELECT 
    user_id AS id,
    first_name || ' ' || last_name AS full_name,  -- Concatenation
    EXTRACT(YEAR FROM birth_date) AS birth_year,
    salary * 1.1 AS salary_with_bonus
FROM employees
LIMIT 5;
```

> For `||`; If any of the operands (the strings being joined) is `NULL`, the entire result of the `||` operation will be `NULL`

**Performance tip:** Always select only needed columns. `SELECT *` in production code:
- Transfers unnecessary data over network
- Prevents use of covering indexes
- Breaks code when schema changes

2. **`DISTINCT` vs `GROUP BY`**

```sql
-- DISTINCT: Simple deduplication
SELECT DISTINCT customer_id 
FROM orders;

-- GROUP BY: Deduplication + aggregation capability
SELECT customer_id, COUNT(*) as order_count
FROM orders
GROUP BY customer_id;
```

**Performance**: `DISTINCT` is faster for simple deduplication
- Use `GROUP BY` when you need aggregations
**Common pitfall:** `DISTINCT` applies to all selected columns

```sql
-- Returns distinct combinations of (customer_id, order_date)
SELECT DISTINCT customer_id, order_date FROM orders;

-- If you want distinct customers who ordered, use:
SELECT DISTINCT customer_id FROM orders;
```

3. **Filtering with `WHERE`**

```sql
-- Numeric comparisons
SELECT * FROM products WHERE price > 100 AND price <= 500;
SELECT * FROM products WHERE price BETWEEN 100 AND 500;  -- Inclusive

-- String matching
SELECT * FROM users WHERE name LIKE 'A%';        -- Starts with A
SELECT * FROM users WHERE name LIKE '%son';      -- Ends with son
SELECT * FROM users WHERE name LIKE '%and%';     -- Contains 'and'
SELECT * FROM users WHERE name LIKE 'J_hn';      -- J_hn (4 chars: John, Jahn)

-- Case-insensitive matching (PostgreSQL)
SELECT * FROM users WHERE name ILIKE '%alice%';

-- Multiple values
SELECT * FROM orders WHERE status IN ('pending', 'processing', 'shipped');

-- NULL handling
SELECT * FROM users WHERE phone_number IS NULL;
SELECT * FROM users WHERE phone_number IS NOT NULL;

-- Date filtering
SELECT * FROM orders 
WHERE order_date >= '2024-01-01' 
  AND order_date < '2024-02-01';
```

**Performance tips:**

- ✅ Use indexed columns in WHERE clauses
- ❌ Avoid functions on indexed columns: `WHERE YEAR(date) = 2024` (no index)
- ✅ Rewrite as: `WHERE date >= '2024-01-01' AND date < '2025-01-01'` (uses index)

4. **Sorting with ORDER BY**

```sql
-- Single column sort
SELECT * FROM products ORDER BY price DESC;

-- Multiple columns (hierarchical)
SELECT * FROM orders 
ORDER BY customer_id ASC, order_date DESC;

-- Sort by computed column
SELECT 
    product_name,
    price,
    quantity,
    price * quantity AS total_value
FROM order_items
ORDER BY total_value DESC;

-- NULL handling
SELECT * FROM users 
ORDER BY last_login_date DESC NULLS LAST;  -- NULLs at end

-- Sort by position (avoid in production, use explicit names)
SELECT name, age FROM users ORDER BY 2 DESC;  -- Orders by age
```

**Performance tips:**
- ✅ Create indexes on frequently sorted columns
- ✅ Use `LIMIT` with `ORDER BY` to avoid sorting entire table
- ❌ Avoid `ORDER BY RANDOM()` for large tables (full scan + expensive sort)

5. **Pagination with LIMIT and OFFSET**

```sql
-- First page (10 results)
SELECT * FROM products ORDER BY product_id LIMIT 10;

-- Second page
SELECT * FROM products ORDER BY product_id LIMIT 10 OFFSET 10;

-- Third page
SELECT * FROM products ORDER BY product_id LIMIT 10 OFFSET 20;
```

**⚠️ Performance degrades with large OFFSETs** (database must scan and discard rows)

**Better approach: Keyset pagination**

```sql
-- Page 1
SELECT * FROM products 
ORDER BY product_id 
LIMIT 10;
-- Last product_id from page 1: 110

-- Page 2 (use last ID from previous page)
SELECT * FROM products 
WHERE product_id > 110 
ORDER BY product_id 
LIMIT 10;
-- Much faster! Uses index directly, no offset scanning
```

### Aggregation Functions

```sql
-- Basic aggregations
SELECT 
    COUNT(*) AS total_orders,                    -- All rows
    COUNT(DISTINCT customer_id) AS unique_customers,
    SUM(amount) AS total_revenue,
    AVG(amount) AS avg_order_value,
    MIN(order_date) AS first_order,
    MAX(order_date) AS last_order,
    STDDEV(amount) AS revenue_std_dev,           -- Standard deviation
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) AS median_order  -- Median
FROM orders;

-- Conditional aggregations
SELECT 
    COUNT(*) AS total_orders,
    COUNT(CASE WHEN status = 'completed' THEN 1 END) AS completed_orders,
    SUM(CASE WHEN status = 'completed' THEN amount ELSE 0 END) AS completed_revenue,
    AVG(CASE WHEN category = 'electronics' THEN price END) AS avg_electronics_price
FROM orders;

-- Aggregation with FILTER (PostgreSQL 9.4+, more readable)
SELECT 
    COUNT(*) AS total_orders,
    COUNT(*) FILTER (WHERE status = 'completed') AS completed_orders,
    SUM(amount) FILTER (WHERE status = 'completed') AS completed_revenue
FROM orders;
```

**NULL behavior in aggregations:**
```sql
-- AVG ignores NULLs
SELECT AVG(rating) FROM reviews;  -- NULL ratings not counted

-- To include NULLs as 0:
SELECT AVG(COALESCE(rating, 0)) FROM reviews;

-- COUNT(*) counts all rows, COUNT(column) counts non-NULL values
SELECT 
    COUNT(*) AS total_users,           -- 1000
    COUNT(phone) AS users_with_phone   -- 850 (150 NULLs)
FROM users;
```

### GROUP BY and HAVING

```sql
-- Basic grouping
SELECT 
    customer_id,
    COUNT(*) AS order_count,
    SUM(amount) AS total_spent,
    AVG(amount) AS avg_order_value
FROM orders
GROUP BY customer_id
ORDER BY total_spent DESC;

-- Multiple columns
SELECT 
    customer_id,
    EXTRACT(YEAR FROM order_date) AS year,
    COUNT(*) AS orders_per_year
FROM orders
GROUP BY customer_id, year
ORDER BY customer_id, year;

-- HAVING: Filter groups (applied after aggregation)
SELECT 
    customer_id,
    COUNT(*) AS order_count,
    SUM(amount) AS total_spent
FROM orders
GROUP BY customer_id
HAVING COUNT(*) >= 5 AND SUM(amount) > 1000  -- High-value customers
ORDER BY total_spent DESC;
```

**WHERE vs HAVING:**
```sql
-- WHERE filters rows before grouping (more efficient)
SELECT category, AVG(price)
FROM products
WHERE price > 10  -- ✅ Filter before grouping
GROUP BY category;

-- HAVING filters groups after aggregation
SELECT category, AVG(price) AS avg_price
FROM products
GROUP BY category
HAVING AVG(price) > 100;  -- Filter groups by aggregate

-- Combined example
SELECT 
    category,
    COUNT(*) AS product_count,
    AVG(price) AS avg_price
FROM products
WHERE in_stock = TRUE           -- Filter individual products
GROUP BY category
HAVING COUNT(*) >= 10           -- Filter categories with 10+ products
ORDER BY avg_price DESC;
```

**Common pitfall:** Selecting non-aggregated columns not in GROUP BY
```sql
-- ❌ Error: name not in GROUP BY
SELECT department, name, COUNT(*)
FROM employees
GROUP BY department;

-- ✅ Correct: Only grouped or aggregated columns
SELECT department, COUNT(*), AVG(salary)
FROM employees
GROUP BY department;

-- ✅ If you need individual names, use window functions instead
SELECT 
    department, 
    name, 
    COUNT(*) OVER (PARTITION BY department) AS dept_count
FROM employees;
```

---

## Advanced SELECT Techniques

### Common Table Expressions (CTEs)

CTEs improve query readability and enable recursive queries.

#### Basic CTE

```sql
-- Without CTE: Nested subquery (hard to read)
SELECT customer_id, total_spent
FROM (
    SELECT customer_id, SUM(amount) AS total_spent
    FROM orders
    WHERE order_date >= '2024-01-01'
    GROUP BY customer_id
) AS recent_orders
WHERE total_spent > 1000;

-- With CTE: Much clearer
WITH recent_orders AS (
    SELECT 
        customer_id, 
        SUM(amount) AS total_spent
    FROM orders
    WHERE order_date >= '2024-01-01'
    GROUP BY customer_id
)
SELECT customer_id, total_spent
FROM recent_orders
WHERE total_spent > 1000;
```

#### Multiple CTEs

```sql
-- Chain multiple CTEs for complex transformations
WITH 
-- Step 1: Get active customers
active_customers AS (
    SELECT customer_id
    FROM orders
    WHERE order_date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY customer_id
),
-- Step 2: Calculate their lifetime metrics
customer_metrics AS (
    SELECT 
        o.customer_id,
        COUNT(*) AS total_orders,
        SUM(o.amount) AS lifetime_value,
        MIN(o.order_date) AS first_order_date,
        MAX(o.order_date) AS last_order_date
    FROM orders o
    WHERE o.customer_id IN (SELECT customer_id FROM active_customers)
    GROUP BY o.customer_id
),
-- Step 3: Add RFM features
rfm_features AS (
    SELECT 
        *,
        CURRENT_DATE - last_order_date AS recency_days,
        total_orders AS frequency,
        lifetime_value AS monetary
    FROM customer_metrics
)
-- Final query
SELECT * FROM rfm_features
WHERE frequency >= 5 AND monetary > 1000
ORDER BY recency_days;
```

**When to use CTEs:**
- ✅ Improve readability of complex queries
- ✅ Reference same subquery multiple times
- ✅ Recursive queries (organizational hierarchies, graphs)
- ✅ Debugging (can test each CTE independently)

**Performance note:** CTEs can sometimes be optimization barriers (database may not push predicates into CTE). Use `EXPLAIN ANALYZE` to verify.

---

#### Recursive CTEs

**Example 1: Generate sequence of numbers**
```sql
-- Generate numbers 1 to 100
WITH RECURSIVE numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n < 100
)
SELECT * FROM numbers;
```

**Example 2: Organizational hierarchy**
```sql
-- Employee reporting structure
CREATE TABLE employees (
    emp_id INT,
    name VARCHAR(100),
    manager_id INT
);

-- Find all reports under a manager (including indirect)
WITH RECURSIVE org_chart AS (
    -- Base case: Start with CEO
    SELECT emp_id, name, manager_id, 1 AS level
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- Recursive case: Add direct reports
    SELECT e.emp_id, e.name, e.manager_id, oc.level + 1
    FROM employees e
    JOIN org_chart oc ON e.manager_id = oc.emp_id
)
SELECT 
    REPEAT('  ', level - 1) || name AS org_structure,
    level
FROM org_chart
ORDER BY level, name;
```

**Example 3: Date series generation (useful for time-series)**
```sql
-- Generate all dates in 2024
WITH RECURSIVE date_series AS (
    SELECT DATE '2024-01-01' AS date
    UNION ALL
    SELECT date + INTERVAL '1 day'
    FROM date_series
    WHERE date < DATE '2024-12-31'
)
SELECT * FROM date_series;
```

---

### Subqueries

Subqueries can appear in SELECT, FROM, WHERE, or HAVING clauses.

#### Subquery in WHERE (Filtering)

```sql
-- Customers who placed orders in the last 30 days
SELECT customer_id, name, email
FROM customers
WHERE customer_id IN (
    SELECT DISTINCT customer_id
    FROM orders
    WHERE order_date >= CURRENT_DATE - INTERVAL '30 days'
);

-- Better with EXISTS (stops at first match, faster for large datasets)
SELECT customer_id, name, email
FROM customers c
WHERE EXISTS (
    SELECT 1
    FROM orders o
    WHERE o.customer_id = c.customer_id
      AND o.order_date >= CURRENT_DATE - INTERVAL '30 days'
);
```

**Performance comparison: IN vs EXISTS**
```sql
-- IN: Executes subquery once, returns list
SELECT * FROM customers
WHERE customer_id IN (SELECT customer_id FROM orders);

-- EXISTS: Correlated subquery, checks row-by-row but stops at first match
SELECT * FROM customers c
WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id);

-- Rule of thumb:
-- - Use EXISTS when subquery returns many rows (better short-circuit)
-- - Use IN when subquery returns few rows (simpler execution plan)
```

#### Subquery in SELECT (Scalar Subquery)

```sql
-- Add order count to customer list
SELECT 
    c.customer_id,
    c.name,
    (SELECT COUNT(*) 
     FROM orders o 
     WHERE o.customer_id = c.customer_id) AS order_count,
    (SELECT MAX(order_date) 
     FROM orders o 
     WHERE o.customer_id = c.customer_id) AS last_order_date
FROM customers c;

-- ⚠️ Performance warning: This runs a subquery for EACH row
-- Better approach: Use JOIN or window function
SELECT 
    c.customer_id,
    c.name,
    COUNT(o.order_id) AS order_count,
    MAX(o.order_date) AS last_order_date
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name;
```

#### Subquery in FROM (Derived Table)

```sql
-- Get top customers by spend, then filter
SELECT customer_id, total_spent, order_count
FROM (
    SELECT 
        customer_id,
        SUM(amount) AS total_spent,
        COUNT(*) AS order_count
    FROM orders
    GROUP BY customer_id
) AS customer_summary
WHERE total_spent > 10000
ORDER BY total_spent DESC;
```

**Subquery vs CTE preference:**
```sql
-- ❌ Hard to read: Nested subqueries
SELECT * FROM (
    SELECT * FROM (
        SELECT * FROM orders WHERE status = 'completed'
    ) WHERE amount > 100
) WHERE customer_id IN (SELECT customer_id FROM vip_customers);

-- ✅ Readable: CTEs
WITH 
completed_orders AS (SELECT * FROM orders WHERE status = 'completed'),
high_value_orders AS (SELECT * FROM completed_orders WHERE amount > 100)
SELECT * FROM high_value_orders
WHERE customer_id IN (SELECT customer_id FROM vip_customers);
```

---

## Window Functions for Analytics

Window functions perform calculations across a set of rows related to the current row, **without collapsing rows** like GROUP BY.

### Window Function Syntax

```sql
SELECT 
    column1,
    column2,
    WINDOW_FUNCTION() OVER (
        PARTITION BY partition_column  -- Optional: Define groups
        ORDER BY sort_column           -- Optional: Define order within partition
        ROWS/RANGE window_frame        -- Optional: Define row window
    ) AS window_result
FROM table_name;
```

### Core Window Functions

#### 1. ROW_NUMBER, RANK, DENSE_RANK

```sql
-- Example data: Student test scores
SELECT 
    student_id,
    test_name,
    score,
    -- Assign unique sequential numbers (even for ties)
    ROW_NUMBER() OVER (PARTITION BY test_name ORDER BY score DESC) AS row_num,
    
    -- Rank with gaps for ties (1, 2, 2, 4)
    RANK() OVER (PARTITION BY test_name ORDER BY score DESC) AS rank,
    
    -- Rank without gaps for ties (1, 2, 2, 3)
    DENSE_RANK() OVER (PARTITION BY test_name ORDER BY score DESC) AS dense_rank
FROM test_scores
ORDER BY test_name, score DESC;

/*
test_name | student_id | score | row_num | rank | dense_rank
----------|------------|-------|---------|------|------------
Math      | 101        | 95    | 1       | 1    | 1
Math      | 102        | 92    | 2       | 2    | 2
Math      | 103        | 92    | 3       | 2    | 2
Math      | 104        | 88    | 4       | 4    | 3  <-- Note rank jump
*/
```

**Use cases:**
- `ROW_NUMBER()`: Deduplication (get latest record), pagination
- `RANK()`: Leaderboards with gap ranking
- `DENSE_RANK()`: Leaderboards without gaps

**Deduplication pattern:**
```sql
-- Get most recent order per customer
WITH ranked_orders AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) AS rn
    FROM orders
)
SELECT * FROM ranked_orders WHERE rn = 1;
```

---

#### 2. LAG and LEAD (Access Previous/Next Rows)

```sql
-- Calculate day-over-day revenue change
SELECT 
    date,
    revenue,
    LAG(revenue, 1) OVER (ORDER BY date) AS prev_day_revenue,
    revenue - LAG(revenue, 1) OVER (ORDER BY date) AS revenue_change,
    ROUND(
        100.0 * (revenue - LAG(revenue, 1) OVER (ORDER BY date)) / 
        LAG(revenue, 1) OVER (ORDER BY date), 
        2
    ) AS revenue_change_pct,
    LEAD(revenue, 1) OVER (ORDER BY date) AS next_day_revenue
FROM daily_revenue
ORDER BY date;
```

**Use cases:**
- Time-series analysis (compare to previous period)
- Churn detection (time since last activity)
- Customer journey analysis (next action prediction)

**Advanced: Compare to same day last year**
```sql
SELECT 
    date,
    revenue,
    LAG(revenue, 365) OVER (ORDER BY date) AS revenue_last_year,
    revenue - LAG(revenue, 365) OVER (ORDER BY date) AS yoy_growth
FROM daily_revenue;
```

---

#### 3. SUM, AVG, COUNT (Running Aggregates)

```sql
-- Running totals and moving averages
SELECT 
    date,
    revenue,
    -- Running total (all rows up to current)
    SUM(revenue) OVER (ORDER BY date) AS cumulative_revenue,
    
    -- 7-day moving average (current + 6 preceding rows)
    AVG(revenue) OVER (
        ORDER BY date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS ma_7day,
    
    -- 30-day moving average
    AVG(revenue) OVER (
        ORDER BY date 
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS ma_30day,
    
    -- Running count
    COUNT(*) OVER (ORDER BY date) AS days_count
FROM daily_revenue
ORDER BY date;
```

**Window frame specifications:**
```sql
-- ROWS: Physical row count
ROWS BETWEEN 6 PRECEDING AND CURRENT ROW       -- Last 7 rows
ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW  -- All rows up to current
ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING       -- 3-row window (prev, current, next)

-- RANGE: Logical range based on values
RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW  -- Last 7 days of data
```

---

#### 4. FIRST_VALUE, LAST_VALUE, NTH_VALUE

```sql
-- Compare each sale to first and last in the month
SELECT 
    date,
    product_id,
    revenue,
    FIRST_VALUE(revenue) OVER (
        PARTITION BY DATE_TRUNC('month', date), product_id
        ORDER BY date
    ) AS first_day_revenue,
    LAST_VALUE(revenue) OVER (
        PARTITION BY DATE_TRUNC('month', date), product_id
        ORDER BY date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING  -- Important!
    ) AS last_day_revenue,
    NTH_VALUE(revenue, 2) OVER (
        PARTITION BY DATE_TRUNC('month', date), product_id
        ORDER BY date
    ) AS second_day_revenue
FROM sales;
```

**⚠️ LAST_VALUE gotcha:** By default, window frame is `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, making `LAST_VALUE` return current row. Always specify full frame for `LAST_VALUE`.

---

#### 5. NTILE (Divide into Buckets)

```sql
-- Divide customers into quartiles by spending
SELECT 
    customer_id,
    total_spent,
    NTILE(4) OVER (ORDER BY total_spent DESC) AS spending_quartile,
    CASE NTILE(4) OVER (ORDER BY total_spent DESC)
        WHEN 1 THEN 'Top 25%'
        WHEN 2 THEN 'Upper Middle'
        WHEN 3 THEN 'Lower Middle'
        WHEN 4 THEN 'Bottom 25%'
    END AS segment
FROM customer_lifetime_value
ORDER BY total_spent DESC;
```

**Use cases:**
- Customer segmentation (RFM quartiles)
- Performance banding
- A/B test group assignment (deterministic bucketing)

---

### Real-World Window Function Examples

#### Example 1: Customer Cohort Retention Analysis

```sql
-- Calculate monthly retention by cohort
WITH user_cohorts AS (
    SELECT 
        user_id,
        DATE_TRUNC('month', registration_date) AS cohort_month
    FROM users
),
user_activity AS (
    SELECT 
        u.user_id,
        u.cohort_month,
        DATE_TRUNC('month', a.activity_date) AS activity_month,
        -- Months since registration
        EXTRACT(YEAR FROM AGE(a.activity_date, u.cohort_month)) * 12 +
        EXTRACT(MONTH FROM AGE(a.activity_date, u.cohort_month)) AS months_since_registration
    FROM user_cohorts u
    JOIN activity_log a ON u.user_id = a.user_id
)
SELECT 
    cohort_month,
    months_since_registration,
    COUNT(DISTINCT user_id) AS active_users,
    FIRST_VALUE(COUNT(DISTINCT user_id)) OVER (
        PARTITION BY cohort_month 
        ORDER BY months_since_registration
    ) AS cohort_size,
    ROUND(
        100.0 * COUNT(DISTINCT user_id) / 
        FIRST_VALUE(COUNT(DISTINCT user_id)) OVER (
            PARTITION BY cohort_month 
            ORDER BY months_since_registration
        ),
        2
    ) AS retention_rate
FROM user_activity
GROUP BY cohort_month, months_since_registration
ORDER BY cohort_month, months_since_registration;
```

---

#### Example 2: Sales Ranking Within Categories

```sql
-- Top 3 products per category by revenue
WITH product_sales AS (
    SELECT 
        p.category,
        p.product_name,
        SUM(oi.quantity * oi.unit_price) AS total_revenue,
        ROW_NUMBER() OVER (
            PARTITION BY p.category 
            ORDER BY SUM(oi.quantity * oi.unit_price) DESC
        ) AS category_rank
    FROM products p
    JOIN order_items oi ON p.product_id = oi.product_id
    GROUP BY p.category, p.product_name
)
SELECT category, product_name, total_revenue, category_rank
FROM product_sales
WHERE category_rank <= 3
ORDER BY category, category_rank;
```

---

#### Example 3: Gap Analysis (Identify Inactive Periods)

```sql
-- Find users with gaps > 30 days between purchases
WITH order_gaps AS (
    SELECT 
        customer_id,
        order_date,
        LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) AS prev_order_date,
        order_date - LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) AS days_since_last_order
    FROM orders
)
SELECT 
    customer_id,
    order_date,
    prev_order_date,
    days_since_last_order
FROM order_gaps
WHERE days_since_last_order > 30
ORDER BY customer_id, order_date;
```

---

## Complex Joins and Set Operations

### Join Types Deep Dive

```sql
-- Sample tables
CREATE TABLE customers (customer_id INT, name VARCHAR(100));
CREATE TABLE orders (order_id INT, customer_id INT, amount DECIMAL);

-- INNER JOIN: Only matching rows
SELECT c.name, o.order_id, o.amount
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id;

-- LEFT JOIN: All customers + matching orders (NULLs for no orders)
SELECT c.name, o.order_id, o.amount
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id;

-- Find customers with NO orders
SELECT c.name
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_id IS NULL;

-- RIGHT JOIN: All orders + matching customers (rarely used, rewrite as LEFT JOIN)
SELECT c.name, o.order_id, o.amount
FROM customers c
RIGHT JOIN orders o ON c.customer_id = o.customer_id;

-- FULL OUTER JOIN: All customers + all orders
SELECT c.name, o.order_id, o.amount
FROM customers c
FULL OUTER JOIN orders o ON c.customer_id = o.customer_id;

-- CROSS JOIN: Cartesian product (every combination)
SELECT c.name, p.product_name
FROM customers c
CROSS JOIN products p;
-- Use case: Generate all possible combinations for recommendation matrix
```

---

### Self-Joins

```sql
-- Find employees and their managers
SELECT 
    e.name AS employee,
    m.name AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.emp_id;

-- Find users who registered on the same day
SELECT 
    u1.name AS user1,
    u2.name AS user2,
    u1.registration_date
FROM users u1
JOIN users u2 
    ON u1.registration_date = u2.registration_date
    AND u1.user_id < u2.user_id  -- Avoid duplicate pairs and self-pairs
ORDER BY u1.registration_date;
```

---

### Multiple Joins

```sql
-- Complex join: Orders with customer, product, and category info
SELECT 
    o.order_id,
    c.customer_name,
    p.product_name,
    cat.category_name,
    oi.quantity,
    oi.unit_price,
    oi.quantity * oi.unit_price AS line_total
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
JOIN categories cat ON p.category_id = cat.category_id
WHERE o.order_date >= '2024-01-01'
ORDER BY o.order_id;
```

**Performance tips for multi-table joins:**
- ✅ Join on indexed columns
- ✅ Filter early (WHERE clauses reduce rows before joins)
- ✅ Start with smallest table
- ✅ Use EXPLAIN ANALYZE to verify join order

---

### Set Operations

#### UNION and UNION ALL

```sql
-- Combine results from multiple queries
-- UNION: Removes duplicates (slower, requires sort)
SELECT customer_id, 'Premium' AS segment FROM premium_customers
UNION
SELECT customer_id, 'Standard' AS segment FROM standard_customers;

-- UNION ALL: Keeps duplicates (faster, no deduplication)
SELECT customer_id, 'Premium' AS segment
FROM premium_customers
UNION ALL
SELECT customer_id, 'Standard' AS segment FROM standard_customers;

-- ✅ Use UNION ALL when you know there are no duplicates (faster)
-- ✅ Use UNION when deduplication is required
```

**Practical example: Combining data from multiple sources**
```sql
-- Aggregate sales from online and retail stores
SELECT 
    'online' AS channel,
    DATE_TRUNC('month', order_date) AS month,
    SUM(amount) AS revenue
FROM online_orders
GROUP BY month

UNION ALL

SELECT 
    'retail' AS channel,
    DATE_TRUNC('month', sale_date) AS month,
    SUM(total) AS revenue
FROM retail_sales
GROUP BY month

ORDER BY month, channel;
```

---

#### INTERSECT

```sql
-- Find customers who purchased both in 2023 AND 2024
SELECT customer_id FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2023
INTERSECT
SELECT customer_id FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2024;

-- Equivalent with JOIN (often faster)
SELECT DISTINCT o1.customer_id
FROM orders o1
JOIN orders o2 ON o1.customer_id = o2.customer_id
WHERE EXTRACT(YEAR FROM o1.order_date) = 2023
  AND EXTRACT(YEAR FROM o2.order_date) = 2024;
```

---

#### EXCEPT (MINUS in Oracle)

```sql
-- Customers who ordered in 2023 but NOT in 2024 (churned)
SELECT customer_id FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2023
EXCEPT
SELECT customer_id FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2024;

-- Equivalent with LEFT JOIN (often clearer)
SELECT DISTINCT o1.customer_id
FROM orders o1
LEFT JOIN orders o2 
    ON o1.customer_id = o2.customer_id 
    AND EXTRACT(YEAR FROM o2.order_date) = 2024
WHERE EXTRACT(YEAR FROM o1.order_date) = 2023
  AND o2.customer_id IS NULL;
```

---

## Date and Time Operations

### Date Extraction and Formatting

```sql
-- Extract components
SELECT 
    order_date,
    EXTRACT(YEAR FROM order_date) AS year,
    EXTRACT(MONTH FROM order_date) AS month,
    EXTRACT(DAY FROM order_date) AS day,
    EXTRACT(DOW FROM order_date) AS day_of_week,      -- 0=Sunday, 6=Saturday
    EXTRACT(DOY FROM order_date) AS day_of_year,      -- 1-365/366
    EXTRACT(QUARTER FROM order_date) AS quarter,
    EXTRACT(WEEK FROM order_date) AS week_number,
    EXTRACT(HOUR FROM order_timestamp) AS hour
FROM orders;

-- Format dates for display
SELECT 
    order_date,
    TO_CHAR(order_date, 'YYYY-MM-DD') AS iso_date,
    TO_CHAR(order_date, 'Mon DD, YYYY') AS readable_date,
    TO_CHAR(order_date, 'Day') AS day_name,
    TO_CHAR(order_timestamp, 'HH24:MI:SS') AS time_24hr
FROM orders;
```

---

### Date Arithmetic

```sql
-- Add/subtract intervals
SELECT 
    order_date,
    order_date + INTERVAL '7 days' AS next_week,
    order_date - INTERVAL '1 month' AS last_month,
    order_date + INTERVAL '1 year' AS next_year,
    order_date + INTERVAL '2 hours 30 minutes' AS deadline
FROM orders;

-- Calculate age/duration
SELECT 
    order_date,
    CURRENT_DATE - order_date AS days_since_order,
    AGE(CURRENT_DATE, order_date) AS age_interval,           -- Returns INTERVAL
    EXTRACT(YEAR FROM AGE(CURRENT_DATE, order_date)) AS years_since,
    DATE_PART('day', CURRENT_DATE - order_date) AS days_since
FROM orders;

-- Business days calculation (exclude weekends)
SELECT 
    order_date,
    delivery_date,
    delivery_date - order_date AS total_days,
    -- Approximate business days (excludes only weekends, not holidays)
    (delivery_date - order_date) - 
    2 * ((delivery_date - order_date) / 7) AS business_days_approx
FROM orders;
```

---

### Date Truncation (Grouping by Period)

```sql
-- Truncate to different periods
SELECT 
    DATE_TRUNC('day', order_timestamp) AS day,
    DATE_TRUNC('week', order_timestamp) AS week,      -- Monday of the week
    DATE_TRUNC('month', order_timestamp) AS month,    -- First day of month
    DATE_TRUNC('quarter', order_timestamp) AS quarter,
    DATE_TRUNC('year', order_timestamp) AS year,
    COUNT(*) AS order_count,
    SUM(amount) AS revenue
FROM orders
GROUP BY 
    DATE_TRUNC('day', order_timestamp),
    DATE_TRUNC('week', order_timestamp),
    DATE_TRUNC('month', order_timestamp),
    DATE_TRUNC('quarter', order_timestamp),
    DATE_TRUNC('year', order_timestamp);

-- Practical: Monthly revenue trend
SELECT 
    DATE_TRUNC('month', order_date) AS month,
    COUNT(*) AS order_count,
    SUM(amount) AS total_revenue,
    AVG(amount) AS avg_order_value
FROM orders
WHERE order_date >= '2024-01-01'
GROUP BY month
ORDER BY month;
```

---

### Timezone Handling

```sql
-- Store timestamps in UTC, convert for display
SELECT 
    order_timestamp AT TIME ZONE 'UTC' AS utc_time,
    order_timestamp AT TIME ZONE 'America/New_York' AS eastern_time,
    order_timestamp AT TIME ZONE 'Asia/Kolkata' AS india_time
FROM orders;

-- Current timestamp in different formats
SELECT 
    NOW() AS timestamp_with_tz,                    -- 2024-01-15 10:30:00+00
    CURRENT_TIMESTAMP AS same_as_now,
    CURRENT_DATE AS date_only,                     -- 2024-01-15
    CURRENT_TIME AS time_only,                     -- 10:30:00
    LOCALTIMESTAMP AS local_timestamp;             -- Depends on server timezone
```

**Best practice:** Always store timestamps in UTC (`TIMESTAMPTZ` in PostgreSQL), convert to local timezone only for display.

---

### Date Series Generation

```sql
-- Generate all dates in a range (useful for filling gaps)
SELECT generate_series(
    '2024-01-01'::DATE,
    '2024-12-31'::DATE,
    '1 day'::INTERVAL
) AS date;

-- Left join with date series to show missing dates
WITH date_range AS (
    SELECT generate_series(
        '2024-01-01'::DATE,
        '2024-01-31'::DATE,
        '1 day'::INTERVAL
    )::DATE AS date
)
SELECT 
    dr.date,
    COALESCE(COUNT(o.order_id), 0) AS order_count,
    COALESCE(SUM(o.amount), 0) AS revenue
FROM date_range dr
LEFT JOIN orders o ON DATE(o.order_date) = dr.date
GROUP BY dr.date
ORDER BY dr.date;
```

---

### Common Date Patterns for ML/DS

#### Pattern 1: Recency, Frequency, Monetary (RFM) Features

```sql
SELECT 
    customer_id,
    CURRENT_DATE - MAX(order_date) AS recency_days,
    COUNT(*) AS frequency,
    SUM(amount) AS monetary,
    AVG(amount) AS avg_order_value,
    STDDEV(amount) AS order_value_std,
    MIN(order_date) AS first_purchase_date,
    CURRENT_DATE - MIN(order_date) AS customer_age_days
FROM orders
GROUP BY customer_id;
```

---

#### Pattern 2: Cohort Analysis Features

```sql
-- Assign cohort based on first purchase month
WITH cohorts AS (
    SELECT 
        customer_id,
        DATE_TRUNC('month', MIN(order_date)) AS cohort_month
    FROM orders
    GROUP BY customer_id
)
SELECT 
    o.customer_id,
    c.cohort_month,
    EXTRACT(YEAR FROM AGE(o.order_date, c.cohort_month)) * 12 + 
    EXTRACT(MONTH FROM AGE(o.order_date, c.cohort_month)) AS months_since_first_purchase,
    o.order_date,
    o.amount
FROM orders o
JOIN cohorts c ON o.customer_id = c.customer_id;
```

---

#### Pattern 3: Seasonal and Temporal Features

```sql
SELECT 
    order_date,
    EXTRACT(YEAR FROM order_date) AS year,
    EXTRACT(MONTH FROM order_date) AS month,
    EXTRACT(DAY FROM order_date) AS day,
    EXTRACT(DOW FROM order_date) AS day_of_week,
    EXTRACT(HOUR FROM order_timestamp) AS hour_of_day,
    
    -- Boolean flags
    CASE WHEN EXTRACT(DOW FROM order_date) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
    CASE WHEN EXTRACT(MONTH FROM order_date) IN (12, 1, 2) THEN 1 ELSE 0 END AS is_winter,
    
    -- Cyclical encoding (for neural networks)
    SIN(2 * PI() * EXTRACT(DOW FROM order_date) / 7) AS day_of_week_sin,
    COS(2 * PI() * EXTRACT(DOW FROM order_date) / 7) AS day_of_week_cos,
    
    -- Time since reference point (days since project start)
    order_date - DATE '2024-01-01' AS days_since_project_start
FROM orders;
```

---

## String Manipulation

### Basic String Functions

```sql
-- Concatenation
SELECT 
    first_name || ' ' || last_name AS full_name,
    CONCAT(first_name, ' ', last_name) AS full_name_alt,  -- Handles NULLs better
    CONCAT_WS(' ', first_name, middle_name, last_name) AS full_name_with_separator
FROM users;

-- Case conversion
SELECT 
    UPPER(name) AS uppercase,
    LOWER(name) AS lowercase,
    INITCAP(name) AS title_case  -- 'john doe' -> 'John Doe'
FROM users;

-- Trimming whitespace
SELECT 
    TRIM(name) AS trimmed,
    LTRIM(name) AS left_trim,
    RTRIM(name) AS right_trim,
    TRIM(BOTH ' ' FROM name) AS custom_trim
FROM users;

-- Length
SELECT 
    name,
    LENGTH(name) AS char_count,
    CHAR_LENGTH(name) AS same_as_length,
    OCTET_LENGTH(name) AS byte_count  -- Different for multi-byte chars
FROM users;
```

---

### Substring and Pattern Matching

```sql
-- Substring extraction
SELECT 
    email,
    SUBSTRING(email FROM 1 FOR POSITION('@' IN email) - 1) AS username,
    SUBSTRING(email FROM POSITION('@' IN email) + 1) AS domain,
    LEFT(email, 5) AS first_5_chars,
    RIGHT(email, 10) AS last_10_chars
FROM users;

-- Pattern matching with LIKE/ILIKE
SELECT * FROM products WHERE name LIKE 'iPhone%';           -- Starts with
SELECT * FROM products WHERE name LIKE '%Pro';              -- Ends with
SELECT * FROM products WHERE name LIKE '%Apple%';           -- Contains
SELECT * FROM products WHERE name ILIKE '%IPHONE%';         -- Case-insensitive

-- Regular expressions (more powerful)
SELECT * FROM products WHERE name ~ '^iPhone [0-9]+';       -- Regex match
SELECT * FROM products WHERE name ~* '^iphone';             -- Case-insensitive regex
SELECT * FROM emails WHERE email ~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$';
```

---

### String Replacement and Manipulation

```sql
-- Replace
SELECT 
    REPLACE(description, 'old', 'new') AS updated_description,
    REGEXP_REPLACE(phone, '[^0-9]', '', 'g') AS digits_only  -- Remove non-digits
FROM products;

-- Split string into array
SELECT 
    SPLIT_PART('apple,banana,cherry', ',', 1) AS first_item,   -- 'apple'
    SPLIT_PART('apple,banana,cherry', ',', 2) AS second_item,  -- 'banana'
    STRING_TO_ARRAY('apple,banana,cherry', ',') AS items_array
FROM products;

-- Padding
SELECT 
    LPAD(order_id::TEXT, 8, '0') AS padded_order_id,  -- '00012345'
    RPAD(name, 20, '.') AS padded_name
FROM orders;
```

---

### Text Processing for ML

```sql
-- Clean and normalize text for NLP
SELECT 
    product_description,
    -- Remove special characters
    REGEXP_REPLACE(product_description, '[^a-zA-Z0-9\s]', '', 'g') AS cleaned,
    -- Convert to lowercase
    LOWER(REGEXP_REPLACE(product_description, '[^a-zA-Z0-9\s]', '', 'g')) AS normalized,
    -- Word count
    ARRAY_LENGTH(STRING_TO_ARRAY(TRIM(product_description), ' '), 1) AS word_count,
    -- Extract hashtags
    REGEXP_MATCHES(product_description, '#\w+', 'g') AS hashtags
FROM products;

-- Fuzzy matching (Levenshtein distance, requires pg_trgm extension)
SELECT 
    a.name AS name_a,
    b.name AS name_b,
    LEVENSHTEIN(a.name, b.name) AS edit_distance,
    SIMILARITY(a.name, b.name) AS similarity_score
FROM products a
CROSS JOIN products b
WHERE a.product_id < b.product_id
  AND SIMILARITY(a.name, b.name) > 0.5
ORDER BY similarity_score DESC;
```

---

## Conditional Logic with CASE

### Simple CASE Expressions

```sql
-- Category mapping
SELECT 
    user_id,
    total_spent,
    CASE 
        WHEN total_spent >= 10000 THEN 'VIP'
        WHEN total_spent >= 1000 THEN 'Gold'
        WHEN total_spent >= 100 THEN 'Silver'
        ELSE 'Bronze'
    END AS customer_tier
FROM customer_lifetime_value;

-- Binary flags
SELECT 
    product_id,
    price,
    CASE WHEN price > 1000 THEN 1 ELSE 0 END AS is_premium,
    CASE WHEN stock_quantity = 0 THEN 1 ELSE 0 END AS is_out_of_stock
FROM products;
```

---

### CASE in Aggregations (Pivot/Conditional Aggregation)

```sql
-- Count orders by status (pivot)
SELECT 
    DATE_TRUNC('month', order_date) AS month,
    COUNT(*) AS total_orders,
    COUNT(CASE WHEN status = 'completed' THEN 1 END) AS completed,
    COUNT(CASE WHEN status = 'pending' THEN 1 END) AS pending,
    COUNT(CASE WHEN status = 'cancelled' THEN 1 END) AS cancelled,
    -- Percentages
    ROUND(100.0 * COUNT(CASE WHEN status = 'completed' THEN 1 END) / COUNT(*), 2) AS completion_rate
FROM orders
GROUP BY month
ORDER BY month;

-- Sum revenue by category
SELECT 
    customer_id,
    SUM(CASE WHEN category = 'Electronics' THEN amount ELSE 0 END) AS electronics_spend,
    SUM(CASE WHEN category = 'Clothing' THEN amount ELSE 0 END) AS clothing_spend,
    SUM(CASE WHEN category = 'Food' THEN amount ELSE 0 END) AS food_spend,
    SUM(amount) AS total_spend
FROM orders o
JOIN products p ON o.product_id = p.product_id
GROUP BY customer_id;
```

---

### CASE with NULL Handling

```sql
-- COALESCE: Return first non-NULL value
SELECT 
    customer_id,
    COALESCE(phone, email, 'No contact') AS primary_contact,
    COALESCE(discount_percent, 0) AS discount  -- Default to 0
FROM customers;

-- NULLIF: Return NULL if two values are equal (avoid division by zero)
SELECT 
    product_id,
    revenue,
    order_count,
    revenue / NULLIF(order_count, 0) AS avg_order_value  -- NULL instead of error
FROM product_stats;

-- Complex NULL handling
SELECT 
    customer_id,
    CASE 
        WHEN phone IS NOT NULL AND email IS NOT NULL THEN 'Both'
        WHEN phone IS NOT NULL THEN 'Phone only'
        WHEN email IS NOT NULL THEN 'Email only'
        ELSE 'No contact'
    END AS contact_status
FROM customers;
```

---

## Advanced Aggregations

### GROUPING SETS, ROLLUP, CUBE

These provide multiple levels of aggregation in a single query.

#### GROUPING SETS

```sql
-- Multiple GROUP BY combinations in one query
SELECT 
    category,
    region,
    SUM(sales) AS total_sales
FROM sales_data
GROUP BY GROUPING SETS (
    (category, region),  -- By category and region
    (category),          -- By category only
    (region),            -- By region only
    ()                   -- Grand total
);

-- Equivalent to UNION ALL of multiple queries:
-- SELECT category, region, SUM(sales) FROM sales_data GROUP BY category, region
-- UNION ALL
-- SELECT category, NULL, SUM(sales) FROM sales_data GROUP BY category
-- UNION ALL
-- SELECT NULL, region, SUM(sales) FROM sales_data GROUP BY region
-- UNION ALL
-- SELECT NULL, NULL, SUM(sales) FROM sales_data
```

---

#### ROLLUP (Hierarchical Aggregation)

```sql
-- Hierarchical totals: year -> quarter -> month
SELECT 
    EXTRACT(YEAR FROM order_date) AS year,
    EXTRACT(QUARTER FROM order_date) AS quarter,
    EXTRACT(MONTH FROM order_date) AS month,
    SUM(amount) AS revenue
FROM orders
GROUP BY ROLLUP(year, quarter, month)
ORDER BY year, quarter, month;

/*
Returns:
year | quarter | month | revenue
-----|---------|-------|--------
2024 | 1       | 1     | 10000   -- Jan 2024
2024 | 1       | 2     | 12000   -- Feb 2024
2024 | 1       | 3     | 11000   -- Mar 2024
2024 | 1       | NULL  | 33000   -- Q1 2024 total
2024 | 2       | 4     | 13000   -- Apr 2024
...
2024 | NULL    | NULL  | 150000  -- 2024 total
NULL | NULL    | NULL  | 300000  -- Grand total
*/
```

---

#### CUBE (All Combinations)

```sql
-- All possible combinations of dimensions
SELECT 
    category,
    region,
    SUM(sales) AS total_sales
FROM sales_data
GROUP BY CUBE(category, region);

/*
Returns aggregations for:
- (category, region)
- (category)
- (region)
- () -- grand total
*/
```

**Use GROUPING() to identify which level:**
```sql
SELECT 
    CASE WHEN GROUPING(category) = 1 THEN 'All Categories' ELSE category END AS category,
    CASE WHEN GROUPING(region) = 1 THEN 'All Regions' ELSE region END AS region,
    SUM(sales) AS total_sales
FROM sales_data
GROUP BY CUBE(category, region);
```

---

### FILTER Clause (PostgreSQL 9.4+)

More readable than CASE for conditional aggregations:

```sql
-- Instead of CASE in aggregation
SELECT 
    category,
    COUNT(*) AS total_products,
    COUNT(*) FILTER (WHERE price > 1000) AS premium_products,
    AVG(price) FILTER (WHERE in_stock = TRUE) AS avg_price_in_stock,
    SUM(revenue) FILTER (WHERE region = 'North America') AS na_revenue
FROM products
GROUP BY category;
```

---

### Statistical Aggregations

```sql
SELECT 
    category,
    COUNT(*) AS n,
    AVG(price) AS mean,
    STDDEV(price) AS std_dev,
    VARIANCE(price) AS variance,
    MIN(price) AS min_price,
    MAX(price) AS max_price,
    
    -- Percentiles
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY price) AS p25,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY price) AS median,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY price) AS p75,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY price) AS p90,
    
    -- Mode (most frequent value)
    MODE() WITHIN GROUP (ORDER BY price) AS mode,
    
    -- Correlation (requires two numeric columns)
    CORR(price, rating) AS price_rating_correlation
FROM products
GROUP BY category;
```

---

## Query Optimization Techniques

### Understanding EXPLAIN ANALYZE

```sql
-- View query execution plan
EXPLAIN SELECT * FROM orders WHERE customer_id = 123;

-- View actual execution statistics
EXPLAIN ANALYZE 
SELECT o.order_id, c.name, SUM(oi.quantity * oi.price) AS total
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.order_date >= '2024-01-01'
GROUP BY o.order_id, c.name;
```

**Key metrics to look for:**
- **Seq Scan**: Full table scan (slow for large tables) → Add index
- **Index Scan**: Using index (good)
- **Rows**: Estimated vs actual row counts → Update statistics if mismatched
- **Cost**: Optimizer's estimate (lower is better)
- **Execution Time**: Actual time taken

---

### Index Strategies

```sql
-- B-tree index (default, good for equality and range queries)
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_date ON orders(order_date);

-- Composite index (for multi-column WHERE/ORDER BY)
CREATE INDEX idx_orders_customer_date ON orders(customer_id, order_date);

-- Partial index (index only subset of rows)
CREATE INDEX idx_active_orders ON orders(order_id) 
WHERE status IN ('pending', 'processing');

-- Expression index
CREATE INDEX idx_users_lower_email ON users(LOWER(email));

-- Covering index (include extra columns to avoid table lookup)
CREATE INDEX idx_orders_customer_covering 
ON orders(customer_id) 
INCLUDE (order_date, amount);

-- Drop unused indexes
DROP INDEX idx_old_index;
```

**Performance tips:**
- ✅ Index foreign keys used in JOINs
- ✅ Index columns in WHERE, ORDER BY, GROUP BY
- ✅ Use composite indexes for multi-column filters (order matters!)
- ❌ Avoid over-indexing (slows INSERT/UPDATE/DELETE)
- ✅ Monitor index usage: 
```sql
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read
FROM pg_stat_user_indexes
ORDER BY idx_scan;
```

---

### Query Rewriting for Performance

#### Example 1: Replace Subquery with JOIN

```sql
-- ❌ Slow: Correlated subquery for each row
SELECT 
    c.customer_id,
    c.name,
    (SELECT COUNT(*) FROM orders o WHERE o.customer_id = c.customer_id) AS order_count
FROM customers c;

-- ✅ Fast: Single JOIN with GROUP BY
SELECT 
    c.customer_id,
    c.name,
    COUNT(o.order_id) AS order_count
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name;
```

---

#### Example 2: Use EXISTS Instead of IN for Large Subqueries

```sql
-- ❌ Slow: IN with large subquery
SELECT * FROM customers
WHERE customer_id IN (
    SELECT customer_id FROM orders WHERE order_date >= '2024-01-01'
);

-- ✅ Fast: EXISTS (stops at first match)
SELECT * FROM customers c
WHERE EXISTS (
    SELECT 1 FROM orders o 
    WHERE o.customer_id = c.customer_id 
      AND o.order_date >= '2024-01-01'
);
```

---

#### Example 3: Push Filters Into Subqueries

```sql
-- ❌ Slow: Filter after aggregation
SELECT customer_id, total_spent
FROM (
    SELECT customer_id, SUM(amount) AS total_spent
    FROM orders
    GROUP BY customer_id
) subq
WHERE customer_id IN (1, 2, 3);

-- ✅ Fast: Filter before aggregation
SELECT customer_id, SUM(amount) AS total_spent
FROM orders
WHERE customer_id IN (1, 2, 3)
GROUP BY customer_id;
```

---

## Feature Engineering with SQL

### RFM (Recency, Frequency, Monetary) Features

```sql
WITH customer_rfm AS (
    SELECT 
        customer_id,
        -- Recency: days since last purchase
        CURRENT_DATE - MAX(order_date) AS recency_days,
        -- Frequency: number of orders
        COUNT(*) AS frequency,
        -- Monetary: total spent
        SUM(amount) AS monetary,
        -- Additional metrics
        AVG(amount) AS avg_order_value,
        STDDEV(amount) AS order_value_std,
        MIN(order_date) AS first_purchase_date,
        MAX(order_date) AS last_purchase_date,
        CURRENT_DATE - MIN(order_date) AS customer_lifetime_days
    FROM orders
    GROUP BY customer_id
)
SELECT 
    *,
    -- RFM scores (1-5, where 5 is best)
    NTILE(5) OVER (ORDER BY recency_days ASC) AS recency_score,      -- Lower is better
    NTILE(5) OVER (ORDER BY frequency DESC) AS frequency_score,
    NTILE(5) OVER (ORDER BY monetary DESC) AS monetary_score,
    -- Combined RFM segment
    CASE 
        WHEN NTILE(5) OVER (ORDER BY recency_days ASC) >= 4 
         AND NTILE(5) OVER (ORDER BY frequency DESC) >= 4 
         AND NTILE(5) OVER (ORDER BY monetary DESC) >= 4 THEN 'Champions'
        WHEN NTILE(5) OVER (ORDER BY recency_days ASC) >= 3 
         AND NTILE(5) OVER (ORDER BY frequency DESC) >= 3 THEN 'Loyal Customers'
        WHEN NTILE(5) OVER (ORDER BY recency_days ASC) <= 2 THEN 'At Risk'
        ELSE 'Potential Loyalists'
    END AS rfm_segment
FROM customer_rfm;
```

---

### Time-Based Aggregation Features

```sql
-- Rolling window features for each customer
SELECT 
    customer_id,
    order_date,
    amount,
    -- Last N orders
    COUNT(*) OVER w AS orders_last_30d,
    SUM(amount) OVER w AS revenue_last_30d,
    AVG(amount) OVER w AS avg_order_value_30d,
    -- Days since last order
    order_date - LAG(order_date, 1) OVER (PARTITION BY customer_id ORDER BY order_date) AS days_since_last_order,
    -- Order frequency (orders per day)
    COUNT(*) OVER w / 30.0 AS orders_per_day
FROM orders
WINDOW w AS (
    PARTITION BY customer_id 
    ORDER BY order_date 
    RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
);
```

---

### Category Preference Features

```sql
-- Customer's purchase distribution across categories
WITH customer_category_spend AS (
    SELECT 
        o.customer_id,
        p.category,
        SUM(o.amount) AS category_spend,
        COUNT(*) AS category_order_count
    FROM orders o
    JOIN products p ON o.product_id = p.product_id
    GROUP BY o.customer_id, p.category
),
customer_total_spend AS (
    SELECT 
        customer_id,
        SUM(category_spend) AS total_spend
    FROM customer_category_spend
    GROUP BY customer_id
)
SELECT 
    ccs.customer_id,
    ccs.category,
    ccs.category_spend,
    ccs.category_order_count,
    cts.total_spend,
    ROUND(100.0 * ccs.category_spend / cts.total_spend, 2) AS category_spend_pct,
    -- Entropy (measure of diversity)
    -SUM(
        (ccs.category_spend / cts.total_spend) * 
        LOG(ccs.category_spend / cts.total_spend)
    ) OVER (PARTITION BY ccs.customer_id) AS purchase_diversity_entropy
FROM customer_category_spend ccs
JOIN customer_total_spend cts ON ccs.customer_id = cts.customer_id;
```

---

### Interaction Features

```sql
-- Product co-purchase patterns
SELECT 
    oi1.product_id AS product_a,
    oi2.product_id AS product_b,
    COUNT(DISTINCT oi1.order_id) AS co_purchase_count,
    COUNT(DISTINCT oi1.order_id) / (
        SELECT COUNT(DISTINCT order_id)::FLOAT 
        FROM order_items 
        WHERE product_id = oi1.product_id
    ) AS co_purchase_rate
FROM order_items oi1
JOIN order_items oi2 
    ON oi1.order_id = oi2.order_id 
    AND oi1.product_id < oi2.product_id  -- Avoid duplicates
GROUP BY oi1.product_id, oi2.product_id
HAVING COUNT(DISTINCT oi1.order_id) >= 10  -- Minimum support
ORDER BY co_purchase_count DESC;
```

---

## Time-Series Analysis Patterns

### Gap Filling with Date Series

```sql
-- Fill missing dates with zeros
WITH date_range AS (
    SELECT generate_series(
        (SELECT MIN(order_date) FROM orders),
        (SELECT MAX(order_date) FROM orders),
        '1 day'::INTERVAL
    )::DATE AS date
)
SELECT 
    dr.date,
    COALESCE(COUNT(o.order_id), 0) AS order_count,
    COALESCE(SUM(o.amount), 0) AS revenue,
    -- Flag missing days
    CASE WHEN COUNT(o.order_id) = 0 THEN 1 ELSE 0 END AS is_missing
    FROM date_range dr
LEFT JOIN orders o ON dr.date = o.order_date
GROUP BY dr.date
ORDER BY dr.date;
```

---

### Moving Averages and Smoothing

```sql
-- Various moving average calculations
SELECT 
    date,
    revenue,
    -- Simple moving average (SMA)
    AVG(revenue) OVER (
        ORDER BY date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS sma_7day,
    
    -- Exponential weighted moving average (approximation)
    AVG(revenue) OVER (
        ORDER BY date 
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS sma_30day,
    
    -- Centered moving average (for smoothing, not forecasting)
    AVG(revenue) OVER (
        ORDER BY date 
        ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING
    ) AS centered_ma_7day,
    
    -- Cumulative moving average
    AVG(revenue) OVER (
        ORDER BY date 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_avg
FROM daily_revenue
ORDER BY date;
```

---

### Trend Detection and Seasonality

```sql
-- Detect trends using linear regression approximation
WITH daily_metrics AS (
    SELECT 
        date,
        revenue,
        ROW_NUMBER() OVER (ORDER BY date) AS day_num
    FROM daily_revenue
),
regression_stats AS (
    SELECT 
        COUNT(*) AS n,
        SUM(day_num) AS sum_x,
        SUM(revenue) AS sum_y,
        SUM(day_num * revenue) AS sum_xy,
        SUM(day_num * day_num) AS sum_x2
    FROM daily_metrics
)
SELECT 
    dm.date,
    dm.revenue,
    -- Linear trend line: y = mx + b
    ((rs.n * rs.sum_xy - rs.sum_x * rs.sum_y) / 
     (rs.n * rs.sum_x2 - rs.sum_x * rs.sum_x)) * dm.day_num +
    (rs.sum_y - ((rs.n * rs.sum_xy - rs.sum_x * rs.sum_y) / 
     (rs.n * rs.sum_x2 - rs.sum_x * rs.sum_x)) * rs.sum_x) / rs.n AS trend_line,
    -- Detrended values (for seasonality analysis)
    dm.revenue - (
        ((rs.n * rs.sum_xy - rs.sum_x * rs.sum_y) / 
         (rs.n * rs.sum_x2 - rs.sum_x * rs.sum_x)) * dm.day_num +
        (rs.sum_y - ((rs.n * rs.sum_xy - rs.sum_x * rs.sum_y) / 
         (rs.n * rs.sum_x2 - rs.sum_x * rs.sum_x)) * rs.sum_x) / rs.n
    ) AS detrended
FROM daily_metrics dm
CROSS JOIN regression_stats rs
ORDER BY dm.date;

-- Year-over-year comparison
SELECT 
    EXTRACT(DOY FROM order_date) AS day_of_year,
    EXTRACT(YEAR FROM order_date) AS year,
    SUM(amount) AS revenue,
    LAG(SUM(amount)) OVER (
        PARTITION BY EXTRACT(DOY FROM order_date) 
        ORDER BY EXTRACT(YEAR FROM order_date)
    ) AS revenue_last_year,
    SUM(amount) - LAG(SUM(amount)) OVER (
        PARTITION BY EXTRACT(DOY FROM order_date) 
        ORDER BY EXTRACT(YEAR FROM order_date)
    ) AS yoy_change
FROM orders
GROUP BY day_of_year, year
ORDER BY year, day_of_year;
```

---

### Change Point Detection

```sql
-- Identify significant changes in metrics
WITH daily_metrics AS (
    SELECT 
        date,
        revenue,
        AVG(revenue) OVER (
            ORDER BY date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS ma_30day,
        STDDEV(revenue) OVER (
            ORDER BY date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS std_30day
    FROM daily_revenue
)
SELECT 
    date,
    revenue,
    ma_30day,
    std_30day,
    -- Z-score (standard deviations from mean)
    CASE 
        WHEN std_30day > 0 THEN (revenue - ma_30day) / std_30day
        ELSE 0
    END AS z_score,
    -- Flag anomalies (|z| > 2)
    CASE 
        WHEN ABS((revenue - ma_30day) / NULLIF(std_30day, 0)) > 2 THEN 1
        ELSE 0
    END AS is_anomaly
FROM daily_metrics
WHERE std_30day IS NOT NULL
ORDER BY date;
```

---

## Data Quality and Validation

### NULL Detection and Profiling

```sql
-- Profile table for NULL counts and completeness
SELECT 
    'customers' AS table_name,
    COUNT(*) AS total_rows,
    COUNT(customer_id) AS non_null_id,
    COUNT(name) AS non_null_name,
    COUNT(email) AS non_null_email,
    COUNT(phone) AS non_null_phone,
    ROUND(100.0 * COUNT(name) / COUNT(*), 2) AS name_completeness_pct,
    ROUND(100.0 * COUNT(email) / COUNT(*), 2) AS email_completeness_pct,
    ROUND(100.0 * COUNT(phone) / COUNT(*), 2) AS phone_completeness_pct
FROM customers;

-- Identify rows with missing critical fields
SELECT *
FROM customers
WHERE email IS NULL OR phone IS NULL;
```

---

### Duplicate Detection

```sql
-- Find exact duplicates
SELECT 
    email,
    COUNT(*) AS duplicate_count
FROM customers
GROUP BY email
HAVING COUNT(*) > 1;

-- Find near-duplicates (fuzzy matching)
SELECT 
    c1.customer_id AS id1,
    c2.customer_id AS id2,
    c1.name AS name1,
    c2.name AS name2,
    SIMILARITY(c1.name, c2.name) AS name_similarity,
    c1.email,
    c2.email
FROM customers c1
JOIN customers c2 ON c1.customer_id < c2.customer_id
WHERE SIMILARITY(c1.name, c2.name) > 0.8
   OR LOWER(c1.email) = LOWER(c2.email)
ORDER BY name_similarity DESC;
```

---

### Outlier Detection

```sql
-- Identify outliers using IQR method
WITH stats AS (
    SELECT 
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY amount) AS q1,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY amount) AS q3,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY amount) - 
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY amount) AS iqr
    FROM orders
)
SELECT 
    o.order_id,
    o.amount,
    s.q1,
    s.q3,
    s.iqr,
    s.q1 - 1.5 * s.iqr AS lower_bound,
    s.q3 + 1.5 * s.iqr AS upper_bound,
    CASE 
        WHEN o.amount < s.q1 - 1.5 * s.iqr THEN 'Low Outlier'
        WHEN o.amount > s.q3 + 1.5 * s.iqr THEN 'High Outlier'
        ELSE 'Normal'
    END AS outlier_status
FROM orders o
CROSS JOIN stats s
WHERE o.amount < s.q1 - 1.5 * s.iqr 
   OR o.amount > s.q3 + 1.5 * s.iqr
ORDER BY o.amount;
```

---

### Referential Integrity Checks

```sql
-- Find orphaned records (foreign key violations)
-- Orders without matching customers
SELECT o.* 
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE c.customer_id IS NULL;

-- Products referenced in orders that don't exist
SELECT DISTINCT oi.product_id
FROM order_items oi
LEFT JOIN products p ON oi.product_id = p.product_id
WHERE p.product_id IS NULL;
```

---

### Data Range Validation

```sql
-- Validate data ranges and business rules
SELECT 
    'Invalid prices' AS issue,
    COUNT(*) AS count
FROM products
WHERE price <= 0 OR price > 1000000

UNION ALL

SELECT 
    'Future order dates',
    COUNT(*)
FROM orders
WHERE order_date > CURRENT_DATE

UNION ALL

SELECT 
    'Negative quantities',
    COUNT(*)
FROM order_items
WHERE quantity <= 0

UNION ALL

SELECT 
    'Invalid email formats',
    COUNT(*)
FROM customers
WHERE email !~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$';
```

---

## Interview Problems and Solutions

### Problem 1: Second Highest Salary

**Question:** Write a query to find the second highest salary from an `employees` table. If there's no second highest, return NULL.

```sql
-- Solution 1: Using OFFSET
SELECT DISTINCT salary
FROM employees
ORDER BY salary DESC
LIMIT 1 OFFSET 1;

-- Solution 2: Using subquery
SELECT MAX(salary) AS second_highest
FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);

-- Solution 3: Using window function (handles ties better)
WITH ranked_salaries AS (
    SELECT 
        salary,
        DENSE_RANK() OVER (ORDER BY salary DESC) AS rank
    FROM employees
)
SELECT salary AS second_highest
FROM ranked_salaries
WHERE rank = 2
LIMIT 1;
```

---

### Problem 2: Consecutive Numbers

**Question:** Find all numbers that appear at least three times consecutively in a `logs` table with columns `id` and `num`.

```sql
-- Sample data:
-- id | num
-- ---|----
-- 1  | 1
-- 2  | 1
-- 3  | 1
-- 4  | 2
-- 5  | 1

-- Solution: Self-join approach
SELECT DISTINCT l1.num AS ConsecutiveNums
FROM logs l1
JOIN logs l2 ON l1.id = l2.id - 1 AND l1.num = l2.num
JOIN logs l3 ON l2.id = l3.id - 1 AND l2.num = l3.num;

-- Solution: Window function approach (more readable)
WITH consecutive_groups AS (
    SELECT 
        num,
        id,
        id - ROW_NUMBER() OVER (PARTITION BY num ORDER BY id) AS grp
    FROM logs
)
SELECT DISTINCT num AS ConsecutiveNums
FROM consecutive_groups
GROUP BY num, grp
HAVING COUNT(*) >= 3;
```

---

### Problem 3: Department Top Three Salaries

**Question:** Find the top 3 highest-paid employees in each department.

```sql
WITH ranked_employees AS (
    SELECT 
        d.department_name,
        e.employee_name,
        e.salary,
        DENSE_RANK() OVER (PARTITION BY d.department_id ORDER BY e.salary DESC) AS salary_rank
    FROM employees e
    JOIN departments d ON e.department_id = d.department_id
)
SELECT 
    department_name,
    employee_name,
    salary
FROM ranked_employees
WHERE salary_rank <= 3
ORDER BY department_name, salary DESC;
```

---

### Problem 4: Cumulative Sum / Running Total

**Question:** Calculate the cumulative salary expenditure for each department by hire date.

```sql
SELECT 
    department_id,
    employee_name,
    hire_date,
    salary,
    SUM(salary) OVER (
        PARTITION BY department_id 
        ORDER BY hire_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_salary
FROM employees
ORDER BY department_id, hire_date;
```

---

### Problem 5: User Retention Rate

**Question:** Calculate the Day 1, Day 7, and Day 30 retention rates for users who registered in January 2024.

```sql
WITH jan_users AS (
    SELECT DISTINCT user_id, registration_date
    FROM users
    WHERE registration_date >= '2024-01-01' 
      AND registration_date < '2024-02-01'
),
retention_activity AS (
    SELECT 
        ju.user_id,
        ju.registration_date,
        -- Did user return on Day 1?
        MAX(CASE 
            WHEN a.activity_date = ju.registration_date + INTERVAL '1 day' 
            THEN 1 ELSE 0 
        END) AS returned_day1,
        -- Did user return within 7 days?
        MAX(CASE 
            WHEN a.activity_date BETWEEN ju.registration_date + INTERVAL '1 day' 
                                    AND ju.registration_date + INTERVAL '7 days'
            THEN 1 ELSE 0 
        END) AS returned_day7,
        -- Did user return within 30 days?
        MAX(CASE 
            WHEN a.activity_date BETWEEN ju.registration_date + INTERVAL '1 day' 
                                    AND ju.registration_date + INTERVAL '30 days'
            THEN 1 ELSE 0 
        END) AS returned_day30
    FROM jan_users ju
    LEFT JOIN activity_log a ON ju.user_id = a.user_id
    GROUP BY ju.user_id, ju.registration_date
)
SELECT 
    COUNT(*) AS total_users,
    ROUND(100.0 * SUM(returned_day1) / COUNT(*), 2) AS day1_retention_pct,
    ROUND(100.0 * SUM(returned_day7) / COUNT(*), 2) AS day7_retention_pct,
    ROUND(100.0 * SUM(returned_day30) / COUNT(*), 2) AS day30_retention_pct
FROM retention_activity;
```

---

### Problem 6: Find Median

**Question:** Find the median salary for each department.

```sql
-- PostgreSQL 9.4+
SELECT 
    department_id,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) AS median_salary
FROM employees
GROUP BY department_id;

-- Alternative: Using window functions
WITH ranked_salaries AS (
    SELECT 
        department_id,
        salary,
        ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary) AS rn,
        COUNT(*) OVER (PARTITION BY department_id) AS total_count
    FROM employees
)
SELECT 
    department_id,
    AVG(salary) AS median_salary
FROM ranked_salaries
WHERE rn IN (FLOOR((total_count + 1) / 2.0), CEIL((total_count + 1) / 2.0))
GROUP BY department_id;
```

---

### Problem 7: Friend Recommendations

**Question:** Given a `friendships` table (user1_id, user2_id), recommend friends: users who are not direct friends but have mutual friends.

```sql
-- Friendships are bidirectional
WITH all_friendships AS (
    SELECT user1_id AS user_a, user2_id AS user_b FROM friendships
    UNION ALL
    SELECT user2_id AS user_a, user1_id AS user_b FROM friendships
),
mutual_friends AS (
    SELECT 
        f1.user_a AS user,
        f2.user_b AS recommended_friend,
        COUNT(*) AS mutual_friend_count
    FROM all_friendships f1
    JOIN all_friendships f2 ON f1.user_b = f2.user_a
    WHERE f1.user_a != f2.user_b  -- Not the same person
    GROUP BY f1.user_a, f2.user_b
)
SELECT 
    mf.user,
    mf.recommended_friend,
    mf.mutual_friend_count
FROM mutual_friends mf
WHERE NOT EXISTS (
    -- Exclude direct friends
    SELECT 1 FROM all_friendships af
    WHERE af.user_a = mf.user 
      AND af.user_b = mf.recommended_friend
)
ORDER BY mf.user, mf.mutual_friend_count DESC;
```

---

### Problem 8: Active Users Over Time

**Question:** Calculate the number of Monthly Active Users (MAU) for each month.

```sql
SELECT 
    DATE_TRUNC('month', activity_date) AS month,
    COUNT(DISTINCT user_id) AS mau
FROM activity_log
GROUP BY month
ORDER BY month;

-- With growth rate
WITH monthly_mau AS (
    SELECT 
        DATE_TRUNC('month', activity_date) AS month,
        COUNT(DISTINCT user_id) AS mau
    FROM activity_log
    GROUP BY month
)
SELECT 
    month,
    mau,
    LAG(mau) OVER (ORDER BY month) AS prev_month_mau,
    mau - LAG(mau) OVER (ORDER BY month) AS mau_change,
    ROUND(
        100.0 * (mau - LAG(mau) OVER (ORDER BY month)) / 
        LAG(mau) OVER (ORDER BY month),
        2
    ) AS mau_growth_pct
FROM monthly_mau
ORDER BY month;
```

---

### Problem 9: Pivot Table (Rows to Columns)

**Question:** Transform quarterly sales data from rows to columns.

```sql
-- Input:
-- product | quarter | sales
-- --------|---------|------
-- A       | Q1      | 100
-- A       | Q2      | 150
-- B       | Q1      | 200

-- Output:
-- product | Q1  | Q2  | Q3  | Q4
-- --------|-----|-----|-----|----
-- A       | 100 | 150 | 0   | 0
-- B       | 200 | 0   | 0   | 0

SELECT 
    product,
    SUM(CASE WHEN quarter = 'Q1' THEN sales ELSE 0 END) AS Q1,
    SUM(CASE WHEN quarter = 'Q2' THEN sales ELSE 0 END) AS Q2,
    SUM(CASE WHEN quarter = 'Q3' THEN sales ELSE 0 END) AS Q3,
    SUM(CASE WHEN quarter = 'Q4' THEN sales ELSE 0 END) AS Q4
FROM sales
GROUP BY product;

-- PostgreSQL crosstab extension (cleaner for many columns)
SELECT * FROM crosstab(
    'SELECT product, quarter, sales FROM sales ORDER BY 1, 2',
    'SELECT DISTINCT quarter FROM sales ORDER BY 1'
) AS ct(product TEXT, Q1 INT, Q2 INT, Q3 INT, Q4 INT);
```

---

### Problem 10: Gap and Island Problem

**Question:** Find consecutive date ranges where a user was active (islands) and gaps where they were inactive.

```sql
-- Find islands (consecutive active dates)
WITH date_groups AS (
    SELECT 
        user_id,
        activity_date,
        activity_date - ROW_NUMBER() OVER (
            PARTITION BY user_id 
            ORDER BY activity_date
        )::INTEGER AS grp
    FROM activity_log
    GROUP BY user_id, activity_date
)
SELECT 
    user_id,
    MIN(activity_date) AS island_start,
    MAX(activity_date) AS island_end,
    MAX(activity_date) - MIN(activity_date) + 1 AS island_length_days
FROM date_groups
GROUP BY user_id, grp
ORDER BY user_id, island_start;
```

---

## SQL Quick Reference Cheat Sheet

### SELECT Essentials

```sql
-- Basic query structure
SELECT column1, column2, aggregate_function(column3)
FROM table_name
WHERE condition
GROUP BY column1, column2
HAVING aggregate_condition
ORDER BY column1 DESC
LIMIT 10 OFFSET 20;

-- Aliases
SELECT column1 AS alias1, column2 alias2  -- AS is optional

-- DISTINCT
SELECT DISTINCT column1 FROM table_name;

-- CASE expression
SELECT CASE 
    WHEN condition THEN result1
    WHEN condition2 THEN result2
    ELSE result3
END AS new_column
FROM table_name;
```

---

### Joins

```sql
-- INNER JOIN (only matching rows)
SELECT * FROM t1 INNER JOIN t2 ON t1.id = t2.id;

-- LEFT JOIN (all from left + matching from right)
SELECT * FROM t1 LEFT JOIN t2 ON t1.id = t2.id;

-- Find non-matches with LEFT JOIN
SELECT t1.* FROM t1 LEFT JOIN t2 ON t1.id = t2.id WHERE t2.id IS NULL;

-- FULL OUTER JOIN (all rows from both)
SELECT * FROM t1 FULL OUTER JOIN t2 ON t1.id = t2.id;

-- CROSS JOIN (Cartesian product)
SELECT * FROM t1 CROSS JOIN t2;
```

---

### Aggregations

```sql
-- Basic aggregates
COUNT(*), COUNT(DISTINCT col), SUM(col), AVG(col), MIN(col), MAX(col)

-- With GROUP BY
SELECT category, COUNT(*), AVG(price)
FROM products
GROUP BY category
HAVING COUNT(*) > 10;

-- Conditional aggregation
SELECT 
    COUNT(*) FILTER (WHERE status = 'active') AS active_count,
    SUM(CASE WHEN status = 'active' THEN amount ELSE 0 END) AS active_revenue
FROM orders;
```

---

### Window Functions

```sql
-- Row numbering
ROW_NUMBER() OVER (PARTITION BY col1 ORDER BY col2)
RANK() OVER (ORDER BY col1)
DENSE_RANK() OVER (ORDER BY col1)

-- Access adjacent rows
LAG(col, offset) OVER (PARTITION BY col1 ORDER BY col2)
LEAD(col, offset) OVER (PARTITION BY col1 ORDER BY col2)

-- Aggregates over windows
SUM(col) OVER (PARTITION BY col1 ORDER BY col2 ROWS BETWEEN n PRECEDING AND CURRENT ROW)
AVG(col) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)  -- 7-day MA

-- Buckets
NTILE(4) OVER (ORDER BY col)  -- Quartiles
```

---

### Date/Time Operations

```sql
-- Extract components
EXTRACT(YEAR FROM date_col), EXTRACT(MONTH FROM date_col)

-- Date arithmetic
date_col + INTERVAL '7 days'
CURRENT_DATE - date_col AS days_ago

-- Truncate to period
DATE_TRUNC('month', date_col)
DATE_TRUNC('week', date_col)

-- Generate date series
generate_series('2024-01-01'::DATE, '2024-12-31'::DATE, '1 day'::INTERVAL)
```

---

### String Operations

```sql
-- Concatenation
'first' || ' ' || 'last'
CONCAT(col1, ' ', col2)

-- Case
UPPER(col), LOWER(col), INITCAP(col)

-- Substring
SUBSTRING(col FROM 1 FOR 10)
LEFT(col, 5), RIGHT(col, 5)

-- Pattern matching
col LIKE '%pattern%'
col ~ '^regex$'  -- Regular expression
```

---

### Common Patterns

```sql
-- Deduplication (get latest per group)
WITH ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY id ORDER BY date DESC) AS rn
    FROM table_name
)
SELECT * FROM ranked WHERE rn = 1;

-- Running total
SELECT date, amount,
    SUM(amount) OVER (ORDER BY date) AS running_total
FROM transactions;

-- Pivot
SELECT category,
    SUM(CASE WHEN year = 2023 THEN sales END) AS y2023,
    SUM(CASE WHEN year = 2024 THEN sales END) AS y2024
FROM sales GROUP BY category;

-- Recursive CTE
WITH RECURSIVE cte AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM cte WHERE n < 100
)
SELECT * FROM cte;
```

---

### Performance Tips

```sql
-- Use EXPLAIN ANALYZE
EXPLAIN ANALYZE SELECT ...;

-- Index usage
CREATE INDEX idx_name ON table(column);
CREATE INDEX idx_composite ON table(col1, col2);

-- Avoid in WHERE
WHERE YEAR(date_col) = 2024  -- ❌ No index
WHERE date_col >= '2024-01-01' AND date_col < '2025-01-01'  -- ✅ Uses index

-- Use EXISTS over IN for large subqueries
WHERE EXISTS (SELECT 1 FROM other_table WHERE ...)  -- ✅
WHERE id IN (SELECT id FROM large_table)  -- ❌ Slower

-- Keyset pagination (better than OFFSET)
WHERE (date, id) > ('2024-01-15', 1000) ORDER BY date, id LIMIT 50;
```

---

