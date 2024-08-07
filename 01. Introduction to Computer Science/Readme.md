<h1 align="center" > Introduction to Computer Science </h1>

<p align="center">
    <a href="#theoretical-computer-science"><img src="./img/main_poster.png" alt="Logo" height=380></a>
</p>

---

**Computer science** is the study of computers and computational systems. It is a broad field which includes everything from the _algorithms_ that make up _software_ to how software interacts with _hardware_ to how well software is developed and designed. **Computer scientists** use various mathematical algorithms, coding procedures, and their expert programming skills to study computer processes and develop new software and systems.

We also have **Computer Engineering** which is concerned with the design of computer hardware and of computer-based devices or software interacting tightly with the hardware for embedded systems and computer-based devices.

**Information systems** as a field has to do with applying today’s information technology to solve today’s problems, typically in the area of businesses and other enterprises.

## Historical Context

Originally, computers were built to solve arithmetic problems, but they have since evolved to run the internet, play video games, create artificial brains, and simulate the universe. At the core, all computations boil down to manipulating 0s and 1s.

Alan Turing is considered the father of computer science. He formalized the concept of a **Turing machine**, which is a simple description of a general-purpose computer. A Turing machine consists of an infinitely long tape divided into cells containing symbols, a head that can read and write symbols, a state register that stores the head's state, and a list of possible instructions. This forms the basis of modern computers.

![AI generated image of a turing machine](./img/Turing_machine.jpeg)
> AI generated image; Not an actual Turing machine

## Theoretical Computer Science

Theoretical computer science examines the problems that can be solved using computational models and algorithms, focusing on their efficiency and solvability.

**Computability Theory** : It studies the general properties of computation, including decidability, computability, and Turing completeness. It addresses what problems can be solved on a computer. **More on [computability theory](https://en.wikipedia.org/wiki/Computability_theory)**

**Automata Theory and Formal Language Theory** : Automata theory studies abstract machines (automata) and the computational problems they can solve. It is closely linked to formal language theory, which examines the properties of formal languages and their relationship to automata. **More on [Automata Theory](https://en.wikipedia.org/wiki/Automata_theory) and [Formal Language Theory](https://en.wikipedia.org/wiki/Formal_language)**

![Automata](./img/Automata_theory.png)

**Computational complexity theory** : This field examines how efficiently problems can be solved, considering resources like time and memory.

1. **Time and space complexity**: Measures the time and memory required to solve a problem as a function of input size, using asymptotic notation such as `O(n)`, `Ω(n)`, and `Θ(n)`.

2. **Classes of problems**: Problems are classified into complexity classes based on their difficulty.

    - **P (polynomial time)**: Problems that can be solved in polynomial time.
    
    - **NP (nondeterministic polynomial time)**: Problems for which a solution can be verified in polynomial time. Examples include the Boolean satisfiability problem (SAT), the traveling salesman problem (TSP), and the vertex cover problem.
    
    - **NP-complete**: The hardest problems in NP, to which all other NP problems can be reduced in polynomial time. A polynomial-time algorithm for any NP-complete problem implies P = NP.

    - **NP-hard**: Problems at least as hard as the hardest NP problems but not necessarily in NP.

### Information theory

This field studies the properties of information and how it can be stored, measured, and communicated. Read more about **[Information Theory article on Wikipedia](https://en.wikipedia.org/wiki/Information_theory)**.

![Information theory](./img/information_theory.png)

### Cryptography 

Cryptography involves techniques for secure communication and data protection, designing and analyzing cryptographic algorithms and protocols to ensure data confidentiality, integrity, and authenticity.

![Cryptography](./img/cryptography.png)

### Graph theory 

Graph theory examines the properties and applications of graphs, which model pairwise relationships between objects. It is used in modeling networks, social relationships, and optimization problems.

## Computer Architecture

A typical computer system's architecture would look like this:

![Computer System's architecture](./img/Arch.png)

It will have;

- **Central Processing Unit (CPU)**: The brain of the computer, responsible for executing instructions and performing calculations.

- **Memory**: RAM (Random Access Memory) Stores data and instructions that the CPU needs to access quickly. Then we have secondary storage devices like hard drives and SSDs that store data more permanently.

- **Input/Output (I/O) devices**: These devices allow the computer to interact with the outside world, such as keyboard, mouse, monitors, network card, etc.

### CPU (Central Processing Unit)

Inside the CPU; we find an integrated circuit or Die (or chip) that contains Processor cores, a memory controller, a graphics processor and many other components. The following is a labelled die shot of a 13-th gen Intel Core i9 processor (Raptor Lake) with 24 cores.

![Die shot of a chip](./img/die_shot.png)
> Image link: [Wikipedia](https://upload.wikimedia.org/wikipedia/commons/a/a4/Intel_Core_i9-13900K_Labelled_Die_Shot.jpg)

If we zoom in further, into each individual core, we will find the following components;


we will find a layout of around 44000 transistors physically execute 32-bit instructions. 

For ease of reference; we will will be using the above diagram to explain the components of a CPU.

It is responsible for executing instructions and performing calculations. 

- **CPU Cores**: Modern CPUs have multiple cores, each capable of executing instructions independently. This allows for parallel processing and improved performance.

- **Cache Systems**: Cache is a small, fast memory unit that stores frequently accessed data and instructions to speed up the access.  Modern CPUs typically have a multi-level cache system, usually referred to as L1, L2, and L3 caches. L1 is the smallest and fastest, located closest to the CPU cores.

- **Instruction Pipeline**: CPUs use instruction pipelines to execute multiple instructions simultaneously. Each stage of the pipeline performs a specific task, such as fetching instructions, decoding them, and executing them. This doesn’t reduce the time it takes to complete an individual instruction; instead, it increases the number of instructions that can be processed simultaneously. This leads to a significant increase in overall CPU throughput

- Main memory, often termed random access memory (RAM), allows independent access to cells, unlike mass storage systems (hard disk, SSD, etc) that handle data in large blocks.

The following diagram illustrates the memory hierarchy in a typical computer system.

<p align="center">
    <a href="#cpu-central-processing-unit"><img src="./img/Memory_hierarchy.png" alt="Logo" height=380></a>
</p>



### Memory and Information representation

Information is represented in computers as 0s and 1s, known as `bits`. These bits can be used to represent numbers, letters, pictures, sounds, and more. The smallest unit of information is a bit, and a group of `8 bits` is called a `byte`. A byte can represent 256 different values `(2^8)`. A byte can also represent a single character in the ASCII character set. 

Computers memory is divided into small units called **cells**. Typically each cell holding a **byte**. 

![Computer Memory](./img/computer_memory_diagram.png)

While there's no physical left or right orientation, we often visualize memory cells as linear, with the high-order end on the left and the low-order end on the right. The high-order bit, or most significant bit, is the leftmost bit, and the low-order bit, or least significant bit, is the rightmost bit.

- We use `boolean operations` to manipulate bits. Boolean operations are logical operations that operate on one or more bits and produce a bit as a result. The most common boolean operations are `AND`, `OR`, and `NOT`. These gates can be combined to create more complex arithmetic and logical operations.

- A gate is a device that generates the output of a Boolean operation based on its input values. In modern computers, gates are typically made as small electronic circuits where 0s and 1s are represented by different voltage levels. They serve as the fundamental components upon which computers are built.



### Programming Languages

Programming languages are formal languages used to communicate instructions to a computer. They allow us to write code that can be executed by a computer to perform specific tasks. Programming languages can be classified into several categories based on their design and intended use.

High level overview of programming languages would look like this:

![Software and programming languages](./img/lang.png)

> We have detailed discussion on our programming language of choice, **Python** in the next section.

- **Low-level languages**: These languages are close to the hardware and provide direct control over the computer's resources. They are difficult to read and write but offer high performance and efficiency. Examples include assembly language and machine code.

- **High-level languages**: These languages are designed to be easy to read and write, making them more accessible to programmers. They are further divided into several categories:

    - **Procedural languages**: These languages focus on procedures or functions that perform specific tasks. Examples include C, Pascal, and Fortran.
    
    - **Object-oriented languages**: These languages organize code into objects that interact with each other. Examples include Java, C++, and Python.
    
    - **Functional languages**: These languages treat computation as the evaluation of mathematical functions and avoid changing state or mutable data. Examples include Haskell, Lisp, and ML.
    
    - **Scripting languages**: These languages are designed for automating tasks and are often used for web development, system administration, and data analysis. Examples include JavaScript, Python, and Ruby.

- **Domain-specific languages (DSL's)**: These languages are designed for specific domains or tasks, such as SQL for database queries, HTML for web development, and MATLAB for scientific computing.

- **Compiled vs. interpreted languages**: Programming languages can be compiled or interpreted. Compiled languages are translated into machine code before execution, while interpreted languages are translated into machine code during execution. Then there are languages that use a combination of both techniques. Both compiled and interpreted languages have their advantages and disadvantages and the way they are executed are pretty complex.

- **Syntax and semantics**: Programming languages have syntax and semantics that define the rules for writing valid code and the meaning of that code. Syntax refers to the structure of the code, while semantics refer to its meaning. Syntax errors occur when the code violates the language's rules, while semantic errors occur when the code does not behave as expected.

- **Paradigms**: Programming languages are based on different programming paradigms, such as `imperative`, `declarative`, `functional`, and `object-oriented`. Each paradigm has its own set of concepts and principles for writing code.

Core computer science topics we are going to focus on are;

- [Data structures and algorithms](../03.%20Data%20Structures%20and%20Algorithms/Readme.md)

- [Operating Systems](./Operating_System.md)

- [Computer Networks](./Computer_Networks.md)

- [Databases](../04.%20Database%20Systems%20and%20Technologies/Readme.md)
