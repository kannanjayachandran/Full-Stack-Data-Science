<h1 align="center"> Computer Science Fundamentals </h1>

Computers were originally built to solve arithmetic problems. Over time, they have evolved to power the internet, create artificial intelligence, run video games, and even simulate the universe. At their core, however, all computations reduce to manipulating **0s and 1s**.

**[Computer science](https://en.wikipedia.org/wiki/Computer_science)** is the study of computers and computational systems. It is a broad discipline covering everything from the _algorithms_ that form _software_ to the ways in which software interacts with _hardware_.

---

## [Boolean Logic](https://en.wikipedia.org/wiki/Boolean_algebra)

Boolean logic is a form of algebra where values are either `true` or `false`. It is the foundation of digital circuits, programming, and computer reasoning. The main operations are:

- **AND**: True only if both inputs are true.
 
- **OR**: True if at least one input is true.
 
- **NOT**: Inverts the input (true becomes false, and vice versa).
 
- **XOR (Exclusive OR)**: True if exactly one input is true.
 
- **NAND (NOT AND)**: True unless both inputs are true.
 
- **NOR (NOT OR)**: True only if both inputs are false.

- **XNOR (Equivalence)**: True if both inputs are the same.

---

## [Computer Architecture](https://en.wikipedia.org/wiki/Computer_architecture)

A computer consists of three fundamental parts:

1. **Central Processing Unit (CPU)** – Executes instructions and performs calculations.

2. **Memory** – Stores instructions and data.  
   - **Primary memory (RAM)**: Fast, volatile storage for data needed immediately.

   - **Secondary storage**: Persistent storage such as SSDs or HDDs.
    
3. **Input/Output (I/O) Devices** – Allow interaction with the outside world (e.g., keyboards, monitors, network cards). 

<div align="center">

![Computer Architecture](./img/Arch.png)

</div>

---

### [CPU (Central Processing Unit)](https://en.wikipedia.org/wiki/Central_processing_unit)

The CPU is an [integrated circuit](https://en.wikipedia.org/wiki/Integrated_circuit) that contains multiple subsystems for executing instructions. Its three key components are:

1. **Control Unit (CU)** – Directs and coordinates the activities of the processor.

2. **Arithmetic Logic Unit (ALU)** – Performs arithmetic (add, subtract, multiply, divide) and logical operations (AND, OR, NOT, etc.).

3. **Registers** – Very small, fast storage locations that temporarily hold data and instructions.  

Modern CPUs also include specialized subsystems such as floating-point units (FPUs), cache memory, and integrated graphics.

#### CPU Cores

Modern CPUs are made of multiple **cores**, each capable of independently executing instructions. This enables **parallelism**, improving multitasking and performance. Many CPUs also support **simultaneous multithreading (SMT)** or **hyper-threading (Intel-specific)**, allowing each core to handle multiple threads.

Key parts of a CPU core include:

- **Floating Point Unit (FPU)** – Handles floating-point arithmetic for scientific and graphical workloads.

- **Integer Execution Units** – Perform arithmetic and logical operations on integers.

- **Decode Unit** – Translates high-level instructions into lower-level micro-operations.

- **Out-of-Order Execution & Retirement Units** – Allow instructions to execute out of order to maximize performance, but ensure results are committed in the correct order.

- **Registers** – Fast storage for integers, addresses, and floating-point data.  

- **Cache Memory** – Multi-level system to speed up data access:  
  - **L1 Cache**: Smallest, fastest, located within the core. 

  - **L2 Cache**: Larger, slower, often shared by a few cores.  

  - **L3 Cache**: Largest, slower, shared across the CPU.  

Other subsystems include:

- **Memory Controller** – Manages communication with RAM.

- **Integrated Graphics (iGPU)** – Handles basic graphics tasks. 
 
- **Interconnects** – Enable communication between cores, caches, and memory.

<div align="center">

![Die shot of a chip](./img/die_shot.png)

</div>


> *Above: A labeled die shot of Intel’s 13th Gen Core i9 (Raptor Lake) processor with 24 cores and 32 threads.* 

- Modern CPUs contain **tens of billions of transistors**, each helping execute 64-bit instructions at incredible speeds.

---

### Memory

All computer data is represented as **binary** (0s and 1s), called **bits**.  

- **1 bit** → smallest unit of data (0 or 1).

- **8 bits = 1 byte** → stores one character or up to 256 unique values ($2^8$ or 256 unique binary patterns).

#### Memory Hierarchy

Memory is organized into a hierarchy balancing **speed, size, and cost**:

<div align="center">

![Machine Cycle](./img/Memory_hierarchy.png)

</div>

- **Registers** → Fastest, smallest (inside CPU).

- **CPU Cache** → Extremely fast, but small and expensive.

- **RAM (Main Memory)** → Larger and moderately fast, but volatile.

- **Secondary Storage** → Massive, persistent, but much slower (SSD/HDD).

#### Memory Organization

Computer memory is divided into **cells**, usually 1 byte each. Memory is often visualized linearly:

![Computer Memory](./img/computer_memory_diagram.png)

- **High-Order End (MSB)**: Leftmost, most significant bit.
  
- **Low-Order End (LSB)**: Rightmost, least significant bit.  

Example: `10110010`  
- MSB = `1`  
- LSB = `0`  

---

### The Machine Cycle

The CPU processes instructions in a continuous **machine cycle**:

1. **Fetch** – Retrieve an instruction from memory.  

2. **Decode** – Translate it into operations the CPU can understand.

3. **Execute** – Perform the required calculation or action.
  
4. **Store** – Write the result back to memory or a register.  

<div align="center">

![Machine Cycle](./img/machine_cycle.png)

</div>

---

## Theoretical Computer Science

Theoretical computer science studies the **fundamental principles of computation**—what can be computed, how efficiently, and under what constraints. It provides the mathematical foundations for computer science, influencing everything from programming languages and algorithms to cryptography and artificial intelligence.

Alan Turing, often considered the father of computer science, formalized the concept of a **Turing machine**. A Turing machine is a simplified model of a general-purpose computer. It consists of:

- An **infinitely long tape** divided into cells, each storing a symbol.

- A **head** that can read, write, and move left or right across the tape.

- A **state register** storing the machine’s current state.

- A **finite set of instructions** that determine the machine’s behavior.  

This model allows us to reason about the limits of computation.

---

### Computability Theory

Computability theory explores what problems can or cannot be solved by an algorithm.  

- **Decidability**: A problem is *decidable* if there exists an algorithm that always produces a "yes" or "no" answer for every valid input. Example: determining if a number is prime.

- **Undecidability**: Some problems, like the *Halting Problem* (deciding if a program will halt or run forever), are provably unsolvable by any algorithm.

- **Computability**: Studies whether a problem can be solved in a finite number of steps using any computational model.

- **Turing Completeness**: A system of data manipulation is *Turing complete* if it can simulate a Turing machine (i.e., express any computation). Most modern programming languages are Turing complete.  

---

### Automata Theory and Formal Language Theory

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

### Computational complexity theory

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

### Information theory

Information theory, founded by Claude Shannon, studies how information is measured, transmitted, and compressed.  

Key ideas include:  
- **Entropy** – A measure of uncertainty or information content.

- **Data compression** – Reducing redundancy (e.g., Huffman coding, ZIP). 

- **Channel capacity** – The maximum rate of reliable information transfer over a communication channel. 

![Information theory](./img/information_theory.png)

---

### Cryptography 

Cryptography focuses on **secure communication** and protecting data from adversaries. It involves designing and analyzing algorithms that ensure:  

- **Confidentiality** – Only intended parties can read the data.

- **Integrity** – Data is not tampered with.

- **Authentication** – Verifying identities.

- **Non-repudiation** – Preventing denial of an action (e.g., digital signatures).

Cryptography is built on theoretical concepts such as **number theory**, **complexity theory**, and **information theory**. Modern systems include public-key cryptography (RSA, ECC) and symmetric-key algorithms (AES). 

![Cryptography](./img/cryptography.png)

---

### Graph theory 

Graph theory studies **graphs**—structures made of nodes (vertices) and connections (edges). Graphs model relationships in computer science and real-world problems such as:  

- **Computer networks** – Routers and links.

- **Social networks** – People and their relationships.

- **Optimization** – Shortest path (Dijkstra’s), spanning trees, network flows.
  
- **Scheduling and matching problems**.  

Graph theory connects deeply with complexity theory, as many NP-complete problems are graph-based.

---

### Related Topics

- [Operating Systems](./Operating_System.md)

- [Computer Networks](./Computer_Networks.md)

- [Linux & Git](./Linux_GIT.md)

- [Data structures and algorithms](../03.%20Data%20Structures%20and%20Algorithms/Readme.md)

- [Databases](../04.%20Database%20Systems/Readme.md)

---
