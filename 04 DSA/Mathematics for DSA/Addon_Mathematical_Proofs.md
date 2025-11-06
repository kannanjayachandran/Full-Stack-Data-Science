# Common Types of Mathematical Proofs

Here's an overview of the most common proof techniques:

## 1. Direct Proof

* **Structure:** To prove a conditional statement $P \implies Q$, assume $P$ is true and then use logical deductions, definitions, axioms, and previously proven theorems to show that $Q$ must also be true.

* **When to Use:** When you can directly connect the hypothesis to the conclusion through a chain of logical steps.

* **Example:** Prove that the sum of two even integers is an even integer.

    * **Proof:**
        * Assume $m$ and $n$ are even integers.

        * By the definition of an even integer, $m = 2k$ for some integer $k$, and $n = 2j$ for some integer $j$.

        * Their sum is $m+n = 2k + 2j$.

        * Factor out 2: $m+n = 2(k+j)$.

        * Since $k$ and $j$ are integers, their sum $k+j$ is also an integer. Let's call this new integer $p$, so $p = k+j$. 
        
        * Thus $m+n = 2p$.

        * By the definition of an even integer, any integer that can be expressed in the form $2p$ (where $p$ is an integer) is even. Therefore $m+n$ is an even integer. QED.

## 2. Proof by Contrapositive

* **Structure:** To prove $P \implies Q$, you instead prove its logically equivalent contrapositive statement: $\neg Q \implies \neg P$ (If not Q, then not P).

* **When to Use:** Often useful when $\neg Q$ provides more concrete information to work with than $P$, or when a direct proof seems difficult.

* **Example:** Prove that if $n^2$ is even, then $n$ is even.

    * **Proof (by contrapositive):**
        * The contrapositive statement is: "If $n$ is not even, then $n^2$ is not even" (i.e., "If $n$ is odd, then $n^2$ is odd").

        * Assume $n$ is an odd integer.

        * By the definition of an odd integer, $n = 2k+1$ for some integer $k$.

        * Then we square $n$
            * $n^2 = (2k+1)^2$
            * $n^2 = (2k)^2 + 2(2k + 1) + 1^2$
            * $n^2 = 4k^2 + 4k + 1$
            * Now we factor out $2$ from the fist two terms;
            * $n^2 = 2(2k^2 + 2k) + 1$.
        
        * Let $m = 2k^2 + 2k$. Since $k$ is an integer, the expression $2k^2 + 2k$ will also result in an integer. Therefore $m$ is also an integer.

        * Substituting $m$ back into our equation for $n^2$: 

            * $n^2 = 2m +1$.

        * By the definition of an odd integer, any integer that can be expressed in the form $2m+1$ (where $m$ is an integer) is an odd integer. Therefore, $n^2$ is an odd integer.

        * Thus, we have proven that if $n$ is odd, then $n^2$ is odd.

        * By contraposition, if $n^2$ is even, then $n$ is even. QED.

## 3. Proof by Contradiction (Reductio ad Absurdum)

* **Structure:** To prove a statement $P$, assume its negation $\neg P$ is true. Then, logically derive a contradiction (a statement that is always false, like $R \land \neg R$). Since assuming $\neg P$ leads to a contradiction, $\neg P$ must be false, meaning $P$ must be true.

> $R \land \neg R$; Here $R$ is a statement; it can either be true or false. Negation of $R$ will always be the opposite of what $R$ is. Hence $R \land \neg R$ is a statement that claims that $R$ is true and that $R$ is also false. This is logically impossible.

* **When to Use:** When direct methods seem difficult, and assuming the opposite provides a good starting point for derivations.

* **Example:** Prove that $\sqrt{2}$ is irrational.

    * **Proof (by contradiction):**
        * Assume, for the sake of contradiction, that $\sqrt{2}$ is rational.

        * By the definition of a rational number, if $\sqrt{2}$ is rational, then $\sqrt{2} = \frac{a}{b}$ where $a$ and $b$ are integers, $b \neq 0$, and $\frac{a}{b}$ is in simplest form (meaning $a$ and $b$ have no common factors other than 1).

        * Squaring both sides: $2 = \frac{a^2}{b^2}$

        * So, $2b^2 = a^2$.

        * This implies that $a^2$ is an even number.

        * If $a^2$ is even, then $a$ must also be even (from the previous example, proof by contrapositive).

        * Since $a$ is even, we can write $a = 2k$ for some integer $k$.

        * Substitute $a=2k$ into $2b^2 = a^2$: $2b^2 = (2k)^2 = 4k^2$.

        * Divide by 2: $b^2 = 2k^2$.

        * This implies that $b^2$ is an even number.

        * If $b^2$ is even, then $b$ must also be even.

        * So, we have shown that both $a$ and $b$ are even. This means $a$ and $b$ have a common factor of 2.

        * This contradicts our initial assumption that $\frac{a}{b}$ was in simplest form (i.e., $a$ and $b$ have no common factors other than 1).

        * Since our assumption that $\sqrt{2}$ is rational led to a contradiction, the assumption must be false.

        * Therefore, $\sqrt{2}$ is irrational. QED.

## 4. Proof by Mathematical Induction

* **Structure:** Used to prove statements about natural numbers (or integers starting from some base). It has two main steps:

    1.  **Base Case:** Show that the statement $P(n)$ is true for the smallest value of $n$ (usually $n=0$ or $n=1$).

    2.  **Inductive Step:** Assume that $P(k)$ is true for an arbitrary natural number $k \ge \text{base case}$ (this is the **Inductive Hypothesis**) Then, use this assumption to prove that $P(k+1)$ is also true.

* **When to Use:** For statements that involve a natural number $n$ and assert something holds for *all* such $n$.

* **Example:** Prove that for all positive integers $n$, the sum of the first $n$ odd integers is $n^2$. That is, $1 + 3 + 5 + \dots + (2n-1) = n^2$.

    * **Proof (by induction):**
        * Let $P(n)$ be the statement $1 + 3 + \dots + (2n-1) = n^2$.

        * **Base Case (n=1):**
            * Left side: $1$

            * Right side: $1^2 = 1$

            * Since $1=1$, $P(1)$ is true.

        * **Inductive Step:**
            * Assume $P(k)$ is true for some positive integer $k$. That is, assume $1 + 3 + \dots + (2k-1) = k^2$. (Inductive Hypothesis)

            * Now we need to show that $P(k+1)$ is true, i.e., $1 + 3 + \dots + (2(k+1)-1) = (k+1)^2$.

            * Consider the left side of $P(k+1)$:

                $1 + 3 + \dots + (2k-1) + (2(k+1)-1)$

                $= (1 + 3 + \dots + (2k-1)) + (2k+2-1)$

                $= k^2 + (2k+1)$ (by the Inductive Hypothesis)

                $= k^2 + 2k + 1$

                $= (k+1)^2$

            * This is the right side of $P(k+1)$.

        * Since $P(1)$ is true and $P(k) \implies P(k+1)$, by the principle of mathematical induction, $P(n)$ is true for all positive integers $n$. QED.

## Proof by Cases (Proof by Exhaustion)

* **Structure:** If a statement can be broken down into a finite number of distinct cases, you can prove the statement by showing it holds true for each and every case.

* **When to Use:** When the conditions or variables naturally fall into a limited number of categories.

* **Example:** Prove that for any integer $n$, $n^2 + n$ is even.

    * **Proof (by cases):**
        * **Case 1: $n$ is even.**

            * If $n$ is even, then $n = 2k$ for some integer $k$.

            * Then $n^2 + n$:
                * $n^2 + n = (2k)^2 + 2k $
                
                * $n^2 + n = 4k^2 + 2k $
                
                * Factor out 2 from the expression; $n^2 + n = 2(2k^2 + k)$.

            * Let $m = 2k^2 + k$. Since $k$ is an integer, This expression $2k^2 + k$ will also result in an integer. Therefore $m$ is also an integer.

            * Thus, we have $n^2 + n = 2m$. By the definition of an even integer, any integer that can be expressed in the form $2m$ (where $m$ is an integer) is even. Therefore, $n^2 + n$ is even when $n$ is even.

        * **Case 2: $n$ is odd.**
            * If $n$ is odd, then by definition $n = 2k+1$ for some integer $k$.

            * Now substitute this into the expression $n^2 + n$:
                * $n^2 + n = (2k+1)^2 + (2k+1) $
                
                * $n^2 + n = (4k^2 + 4k + 1) + (2k+1) $
                
                * $n^2 + n = 4k^2 + 6k + 2 $
                
                * $n^2 + n = 2(2k^2 + 3k + 1)$.

            * Let $p = 2k^2 + 3k + 1$. Since k is an integer, the expression $2^k +3 k + 1$ will also always result in an integer. Therefore, p is an integer.

            * Thus, we have $n^2 + n = 2p$. By the definition of an even integer, any integer that can be expressed in the form $2p$ (where p is an integer) is even. Therefore, $n^2 +n$ is even when $n$ is odd.

        * Since all integers are either even or odd, and in both cases $n^2+n$ is even, the statement holds for all integers $n$. QED.

## 6. Proof by Counterexample

* **Structure:** To disprove a universal statement (e.g., "For all X, P(X) is true"), you only need to find one specific example where P(X) is false.

* **When to Use:** To show that a conjecture or statement is false.

* **Example:** Disprove the statement "All prime numbers are odd."

    * **Disproof (by counterexample):**
        * The number 2 is a prime number, but 2 is an even number.

        * Since we found a prime number (2) that is not odd, the statement "All prime numbers are odd" is false. QED.

### Common Logical Fallacies to Avoid

* **Begging the Question (Circular Reasoning):** Assuming the conclusion you are trying to prove as part of your argument.

* **Affirming the Consequent:** Incorrectly assuming that if $Q$ is true, then $P$ must be true, given $P \implies Q$. (e.g., If it rains, the ground is wet. The ground is wet, so it must have rained. - Incorrect, could be sprinklers).

* **Denying the Antecedent:** Incorrectly assuming that if $P$ is false, then $Q$ must be false, given $P \implies Q$. (e.g., If it rains, the ground is wet. It didn't rain, so the ground isn't wet. - Incorrect, again, sprinklers).

* **Confusing Correlation with Causation:** Assuming that because two events happen together, one causes the other.

* **Hasty Generalization:** Drawing a broad conclusion from a very small or unrepresentative sample.

* **Fallacy of the Converse/Inverse:** These are specific cases of affirming the consequent/denying the antecedent.

* **Non Sequitur:** A conclusion that does not logically follow from the premises.
