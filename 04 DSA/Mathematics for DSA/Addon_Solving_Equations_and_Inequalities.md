# Solving Equations and Inequalities

## I. Solving Equations

An equation is a mathematical statement that asserts the equality of two expressions. Solving an equations essentially means, finding the value(s) of the variable(s) that make the equation true. We can use the following operations used to manipulate equations:

- Multiply or divide both sides by the same non-zero number.

- Add or subtract a value from both sides.

- Move a term to the other side by changing its sign (from $+$ to $−$ or vice versa).

### A. Linear Equations

* **Definition:** An equation of the form $ax + b = c$, where $a, b, c$ are constants and $a \neq 0$.

* **Method:** Isolate the variable using inverse operations (addition/subtraction, multiplication/division).

* **Example:** Solve $3x - 5 = 10$

    $3x = 15$ (add 5 to both sides)
    
    $x = 5$ (divide by 3 on both sides)

### B. Quadratic Equations

* **Definition:** An equation of the form $ax^2 + bx + c = 0$, where $a, b, c$ are constants and $a \neq 0$.

* **Methods:**
    1.  **Factoring:** If the quadratic expression can be factored, set each factor to zero and solve.
        * **Example:** Solve $x^2 - 5x + 6 = 0$

            $(x-2)(x-3) = 0$

            $x-2 = 0 \implies x = 2$

            $x-3 = 0 \implies x = 3$

            > For factorizing monic quadratic equation (equations where a = 1 in $ax^2 + bx + c = 0$), we need to find two numbers such that when we multiply them we get $c$ and when we add them we get $b$.

            > For factoring $ax^2 + bx + c = 0$ when $a \neq 1$, we need to find two numbers such that: 1. Their product is $ac$. 2. Their sum is $b$.
            >
            >Once you find these two numbers, say m and n: Rewrite the middle term $bx$ as $mx+nx$. Then factor by grouping
            >
            > Example : $2x^2 + 7x + 3 = 0$
            >
            > $ac$ = 6 and sum is $b$ = 7; numbers are $m$ = 1 and $n$ = 6
            >
            > $2x^2 + 1x + 6x + 3$ = 0
            >
            > $2x^2 + x$ + $6x + 3$ = 0
            >
            > $x(2x + 1)$ + $3(2x + 1)$ = 0
            >
            > $(2x + 1)$ $(x + 3)$  = 0

    2.  **Quadratic Formula:** For any quadratic equation, the solutions are given by $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$.
        * **Example:** Solve $2x^2 + 3x - 1 = 0$

            Here $a=2, b=3, c=-1$.

            $x = \frac{-3 \pm \sqrt{3^2 - 4(2)(-1)}}{2(2)}$

            $x = \frac{-3 \pm \sqrt{9 + 8}}{4}$
            
            $x = \frac{-3 \pm \sqrt{17}}{4}$

    3.  **Completing the Square:** A method to transform the equation into a perfect square trinomial. Useful for deriving the quadratic formula and understanding conic sections.
        * **Example:** Solve $x^2 + 6x - 7 = 0$

            $x^2 + 6x = 7$

            $x^2 + 6x + (6/2)^2 = 7 + (6/2)^2$

            $x^2 + 6x + 9 = 7 + 9$

            $(x+3)^2 = 16$

            $x+3 = \pm 4$

            $x = -3 \pm 4$

            $x = 1$ or $x = -7$

### C. Rational Equations

* **Definition:** Equations involving rational expressions (fractions with variables in the denominator).

* **Method:** Multiply both sides by the least common multiple (LCM) of the denominators to eliminate the fractions. Remember to check for extraneous solutions (values that make the original denominators zero).

* **Example:** Solve $\frac{2}{x+1} = \frac{3}{x-2}$

    Multiply by $(x+1)(x-2)$:

    $2(x-2) = 3(x+1)$

    $2x - 4 = 3x + 3$

    $-7 = x$

    Check: $x=-7$ does not make denominators zero. So $x=-7$ is the solution.

### D. Radical Equations

* **Definition:** Equations where the variable is under a radical sign (square root, cube root, etc.).

* **Method:** Isolate the radical, then raise both sides to the power corresponding to the index of the radical. Remember to check for extraneous solutions.

* **Example:** Solve $\sqrt{x+2} = 3$

    $(\sqrt{x+2})^2 = 3^2$

    $x+2 = 9$

    $x = 7$

    Check: $\sqrt{7+2} = \sqrt{9} = 3$. Solution is valid.

### E. Absolute Value Equations

* **Definition:** Equations involving the absolute value of an expression.

* **Method:** If $|ax+b| = c$ (where $c \ge 0$), then $ax+b = c$ or $ax+b = -c$.

* **Example:** Solve $|2x - 1| = 5$

    $2x - 1 = 5$ or $2x - 1 = -5$

    $2x = 6$ or $2x = -4$

    $x = 3$ or $x = -2$

**F. Exponential Equations**

* **Definition:** Equations where the variable is in the exponent.
* **Method:**
    1.  If bases can be made the same, equate the exponents.
    2.  Use logarithms to bring the exponent down.
* **Example 1 (Same Base):** Solve $2^{x+1} = 8$
    $2^{x+1} = 2^3$
    $x+1 = 3 \implies x = 2$
* **Example 2 (Using Logarithms):** Solve $3^x = 10$
    $\log(3^x) = \log(10)$
    $x \log 3 = 1$
    $x = \frac{1}{\log 3}$

**G. Logarithmic Equations**

* **Definition:** Equations involving logarithms.
* **Method:** Use the properties of logarithms to simplify and then convert to exponential form, or equate arguments if bases are the same. Remember the domain of logarithms (argument must be positive).
* **Example:** Solve $\log_2(x-1) = 3$
    Convert to exponential form: $x-1 = 2^3$
    $x-1 = 8$
    $x = 9$
    Check: $x-1 > 0 \implies 9-1 = 8 > 0$. Solution is valid.

## II. Solving Inequalities

An inequality is a mathematical statement that compares two expressions using symbols like $<, >, \le, \ge$. The goal is to find the range of values for the variable that make the inequality true.

**General Rules:**

* Adding or subtracting the same number to both sides does not change the inequality direction.

* Multiplying or dividing both sides by a **positive** number does not change the inequality direction.

* Multiplying or dividing both sides by a **negative** number **reverses** the inequality direction.

**A. Linear Inequalities**

* **Method:** Similar to linear equations, but pay attention to reversing the inequality sign when multiplying/dividing by a negative number.
* **Example:** Solve $2x - 3 < 7$
    $2x < 10$ (add 3 to both sides)
    $x < 5$ (divide by 2, positive, so sign doesn't change)
    Solution in interval notation: $(-\infty, 5)$

**B. Compound Inequalities**

* **Definition:** Two inequalities joined by "and" or "or".
* **Method:** Solve each inequality separately.
    * **"And":** Find the intersection of the solution sets.
        * **Example:** Solve $-2 \le 3x+1 < 7$
            $-2 \le 3x+1$ and $3x+1 < 7$
            $-3 \le 3x \implies -1 \le x$
            $3x < 6 \implies x < 2$
            Intersection: $[-1, 2)$
    * **"Or":** Find the union of the solution sets.
        * **Example:** Solve $x+3 < 2$ or $2x-1 \ge 5$
            $x < -1$ or $2x \ge 6 \implies x \ge 3$
            Union: $(-\infty, -1) \cup [3, \infty)$

**C. Quadratic Inequalities**

* **Definition:** Inequalities involving a quadratic expression (e.g., $ax^2 + bx + c > 0$).
* **Method (Sign Analysis):**
    1.  Find the roots of the corresponding quadratic equation ($ax^2 + bx + c = 0$). These are the critical points.
    2.  Plot the critical points on a number line. They divide the number line into intervals.
    3.  Choose a test value from each interval and substitute it into the inequality to determine the sign of the quadratic expression in that interval.
    4.  Select the intervals that satisfy the inequality.
* **Example:** Solve $x^2 - x - 6 > 0$
    Roots of $x^2 - x - 6 = 0$ are $(x-3)(x+2) = 0 \implies x=3, x=-2$.
    Critical points: $-2, 3$.
    Intervals: $(-\infty, -2)$, $(-2, 3)$, $(3, \infty)$
    * Test $x=-3$ in $(-\infty, -2)$: $(-3)^2 - (-3) - 6 = 9+3-6 = 6 > 0$. So this interval works.
    * Test $x=0$ in $(-2, 3)$: $0^2 - 0 - 6 = -6 \ngtr 0$. So this interval doesn't work.
    * Test $x=4$ in $(3, \infty)$: $4^2 - 4 - 6 = 16-4-6 = 6 > 0$. So this interval works.
    Solution: $(-\infty, -2) \cup (3, \infty)$

**D. Rational Inequalities**

* **Definition:** Inequalities involving rational expressions.
* **Method (Sign Analysis):**
    1.  Move all terms to one side so that the other side is zero.
    2.  Find the critical points: values that make the numerator zero and values that make the denominator zero.
    3.  Plot the critical points on a number line.
    4.  Choose test values in each interval and substitute them into the inequality to determine the sign.
    5.  Select the intervals that satisfy the inequality. Remember that values making the denominator zero are never part of the solution.
* **Example:** Solve $\frac{x-1}{x+2} \le 0$
    Critical points: $x-1=0 \implies x=1$ (from numerator); $x+2=0 \implies x=-2$ (from denominator).
    Intervals: $(-\infty, -2)$, $(-2, 1]$, $[1, \infty)$
    * Test $x=-3$: $\frac{-3-1}{-3+2} = \frac{-4}{-1} = 4 \not\le 0$.
    * Test $x=0$: $\frac{0-1}{0+2} = \frac{-1}{2} = -0.5 \le 0$. This interval works.
    * Test $x=2$: $\frac{2-1}{2+2} = \frac{1}{4} = 0.25 \not\le 0$.
    Solution: $(-2, 1]$ (Note: $-2$ is excluded because it makes the denominator zero).

**E. Absolute Value Inequalities**

* **Method:**
    * If $|ax+b| < c$ (where $c > 0$), then $-c < ax+b < c$.
    * If $|ax+b| > c$ (where $c > 0$), then $ax+b < -c$ or $ax+b > c$.
* **Example 1 ($<$):** Solve $|x-3| < 2$
    $-2 < x-3 < 2$
    $-2+3 < x < 2+3$
    $1 < x < 5$
    Solution: $(1, 5)$
* **Example 2 ($>$):** Solve $|2x+1| \ge 7$
    $2x+1 \le -7$ or $2x+1 \ge 7$
    $2x \le -8$ or $2x \ge 6$
    $x \le -4$ or $x \ge 3$
    Solution: $(-\infty, -4] \cup [3, \infty)$

**Key Concepts and Tips:**

* **Inverse Operations:** Use addition/subtraction, multiplication/division, exponentiation/roots, and logarithms/exponential to isolate the variable.

* **Checking Solutions:** Always substitute your solutions back into the original equation/inequality, especially for rational, radical, and absolute value problems, to identify extraneous solutions.

* **Number Line:** A powerful tool for visualizing intervals and analyzing signs, particularly for inequalities.

* **Interval Notation:** A concise way to express solution sets for inequalities.

* **Domain Restrictions:** Be mindful of restrictions on variables (e.g., denominators cannot be zero, arguments of logarithms must be positive, expressions under even roots must be non-negative).
