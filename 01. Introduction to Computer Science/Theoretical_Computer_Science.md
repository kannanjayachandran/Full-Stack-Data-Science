<h1 align="center"> Theoretical Computer Science </h1>

Theoretical computer science studies the **fundamental principles of computation**—what can be computed, how efficiently, and under what constraints. It provides the mathematical foundations for computer science, influencing everything from programming languages and algorithms to cryptography and artificial intelligence.

Alan Turing, often considered the father of computer science, formalized the concept of a **Turing machine**. A Turing machine is a simplified model of a general-purpose computer. It consists of:

- An **infinitely long tape** divided into cells, each storing a symbol.

- A **head** that can read, write, and move left or right across the tape.

- A **state register** storing the machine’s current state.

- A **finite set of instructions** that determine the machine’s behavior.  

This model allows us to reason about the limits of computation.

---

## Computability Theory

Computability theory explores what problems can or cannot be solved by an algorithm.  

- **Decidability**: A problem is *decidable* if there exists an algorithm that always produces a "yes" or "no" answer for every valid input. Example: determining if a number is prime.

- **Undecidability**: Some problems, like the *Halting Problem* (deciding if a program will halt or run forever), are provably unsolvable by any algorithm.

- **Computability**: Studies whether a problem can be solved in a finite number of steps using any computational model.

- **Turing Completeness**: A system of data manipulation is *Turing complete* if it can simulate a Turing machine (i.e., express any computation). Most modern programming languages are Turing complete.  

---

## Automata Theory and Formal Language Theory

Automata theory studies **abstract machines** (mathematical models of computation) and the problems they can solve. Examples include:

- **Finite Automata**: Recognize regular languages.

- **Pushdown Automata**: Recognize context-free languages (e.g., used in parsing).

- **Turing Machines**: Recognize recursively enumerable languages.  

Formal language theory focuses on the structure of languages and their classifications via the **Chomsky hierarchy**: 

1. **Regular Languages** – Recognized by finite automata.

2. **Context-Free Languages** – Recognized by pushdown automata.

3. **Context-Sensitive Languages** – Recognized by linear bounded automata.
  
4. **Recursively Enumerable Languages** – Recognized by Turing machines. 

![Automata](./img/Automata_theory.png)

---

## Computational complexity theory

While computability asks *what can be solved*, complexity theory asks *how efficiently* problems can be solved. It studies resource usage like **time** and **space**.

- **Asymptotic complexity**: Uses notations like  
  - `O(n)` → upper bound (worst case).  
  - `Ω(n)` → lower bound (best case).  
  - `Θ(n)` → tight bound (average/typical growth).  

- **Complexity Classes**:  
  - **P** – Problems solvable in polynomial time.

  - **NP** – Problems whose solutions can be verified in polynomial time.

  - **NP-complete** – The hardest problems in NP; if one can be solved in polynomial time, then all NP problems can. Examples: SAT, TSP, Vertex Cover. 
   
  - **NP-hard** – At least as hard as NP-complete problems, but not necessarily in NP (solutions may not be verifiable efficiently).  

The famous **P vs NP problem** asks whether every problem whose solution can be quickly verified (NP) can also be quickly solved (P). This remains one of the greatest unsolved problems in mathematics and computer science.

---

## Information theory

Information theory, founded by Claude Shannon, studies how information is measured, transmitted, and compressed.  

Key ideas include:  
- **Entropy** – A measure of uncertainty or information content.

- **Data compression** – Reducing redundancy (e.g., Huffman coding, ZIP). 

- **Channel capacity** – The maximum rate of reliable information transfer over a communication channel. 

![Information theory](./img/information_theory.png)

---

## Cryptography 

Cryptography focuses on **secure communication** and protecting data from adversaries. It involves designing and analyzing algorithms that ensure:  

- **Confidentiality** – Only intended parties can read the data.

- **Integrity** – Data is not tampered with.

- **Authentication** – Verifying identities.

- **Non-repudiation** – Preventing denial of an action (e.g., digital signatures).

Cryptography is built on theoretical concepts such as **number theory**, **complexity theory**, and **information theory**. Modern systems include public-key cryptography (RSA, ECC) and symmetric-key algorithms (AES). 

![Cryptography](./img/cryptography.png)

---

## Graph theory 

Graph theory studies **graphs**—structures made of nodes (vertices) and connections (edges). Graphs model relationships in computer science and real-world problems such as:  

- **Computer networks** – Routers and links.

- **Social networks** – People and their relationships.

- **Optimization** – Shortest path (Dijkstra’s), spanning trees, network flows.
  
- **Scheduling and matching problems**.  

Graph theory connects deeply with complexity theory, as many NP-complete problems are graph-based.

---

## Summary

Theoretical computer science provides the **mathematical foundation** of computing. By studying computability, automata, complexity, information, cryptography, and graph theory, we gain insights into **what computers can do, what they cannot do, and how efficiently they can do it**.  
