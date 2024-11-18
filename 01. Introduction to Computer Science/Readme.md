<h1 align="center" > Introduction to Computer Science </h1>

- **Computer science** is the study of computers and computational systems. It is a broad field which includes everything from the _algorithms_ that make up _software_ to how software interacts with _hardware_. 

- **Computer scientists** use various mathematical algorithms, coding procedures, and their expert programming skills to study computer processes and develop new software and systems.

- **Computer Engineering** is concerned with the design of computer hardware and software interacting tightly with the hardware for embedded systems and computer-based devices.

Computers were built to solve arithmetic problems, but they have since evolved to run the internet, play video games, create artificial brains, and simulate the universe. At the core, all computations boil down to manipulating 0s and 1s.

## Computer Architecture

A typical computer system's architecture would look like this:

<div align="center">

![Computer Architecture](./img/Arch.png)

</div>

It will have;

- **Central Processing Unit (CPU)**: Responsible for executing instructions and performing calculations (ALU).

- **Memory**: RAM (Random Access Memory) Stores data and instructions that the CPU needs to access quickly. Then we have secondary storage devices like hard drives and SSDs for persistent storage.

- **Input/Output (I/O) devices**: These devices allow the computer to interact with the outside world, such as keyboard, mouse, monitors, network card, etc.

### CPU (Central Processing Unit)

Inside the CPU; we have an IC (integrated circuit) or Die (or chip) that contains the `Processor cores`, a `memory controller`, a `graphics processor` and many other components. 

The following is a labelled die shot of `13-th gen Intel Core i9 processor` (Raptor Lake) with 24 cores.

![Die shot of a chip](./img/die_shot.png)

> Image Courtesy ~ [Wikipedia](https://upload.wikimedia.org/wikipedia/commons/a/a4/Intel_Core_i9-13900K_Labelled_Die_Shot.jpg)

**CPU Cores**: Modern CPUs have multiple cores, each capable of executing instructions independently. Further  each core could execute multiple threads simultaneously. The number of threads can be twice the number of cores in CPUs with hyper-threading(Intel specific; AMD has something known as SMT (Simultaneous Multithreading)). In this particular architecture not all cores have similar performance to efficiency ratios; some cores are optimized for performance while others are optimized for efficiency (P-Cores and E-Cores).

The following are the components typically found inside a modern CPU Core:

- **Floating Point Unit (FPU)**: A specialized unit for performing floating-point arithmetic operations, such as addition, subtraction, multiplication, and division.

- **Integer Execution Units**: These units perform integer arithmetic operations, such as addition, subtraction, multiplication, and division. It is a critical part of the CPU for general purpose computing.

- **Out of order scheduler and Retirement unit**: This unit allows the CPU to execute instructions out of the order they appear in the program. This increases efficiency by making better use of CPU resources. After instructions are executed out-of-order, the retirement unit ensures they are retired (completed) in the correct order, maintaining the program's logical flow.

- **Decode Unit**: This unit decodes instructions fetched from memory into micro-operations that can be executed by the CPU's execution units.

- **Registers (Integer and Floating Point)**: Registers are small, fast storage locations within the CPU used to store data temporarily during processing. They are used to hold operands, intermediate results, and memory addresses.

- **Cache (L1, L2, L3)**: Small, fast memory unit that stores frequently accessed data and instructions to speed up access. Modern CPUs typically have a multi-level cache system, usually referred to as L1, L2, and L3 caches. L1 is the smallest and fastest, located closest to the CPU cores.

- **Branch Prediction Unit**: This unit predicts the outcome of conditional branches in the program to minimize the performance impact of branch mis-predictions.

With the above architecture; inside every CPU core we will find a layout of around 44000 transistors physically execute 32-bit instructions, with a grand total of around 26 Million transistors in the processor.

---

**Let's see what happens when we run a python program that adds two numbers**.

```python
a, b = 5, 3
c = a + b

print(c)
```

**PHASE 1: Python Source Code Execution**

1. We run the program in the terminal. `python3 add.py`

2. The Python interpreter reads the source code and breaks it down into tokens (keywords, operators, identifiers). This process is known as lexical analysis or tokenization.

3. The tokens are parsed to form an Abstract Syntax Tree (AST), representing the structure of the code.

4. The AST is compiled into Python bytecode, an intermediate representation that is platform-independent.

5. The Python Virtual Machine (`PVM`) executes the bytecode. For each bytecode instruction:

    - Fetches the instruction.

    - Decodes the instruction.

    - Executes the instruction using a stack-based virtual machine.

**PHASE 2: CPU Execution**

1. The CPU fetches the next instruction (bytecode) from memory.

2. The decode unit translates the instruction into micro-operations (`μOps`) that the CPU can execute.

3. Instructions are sent to the out-of-order execution unit, allowing the CPU to execute instructions as resources become available rather than strictly sequentially.

4. Integer and floating-point values (e.g., `a` and `b`) are loaded into CPU registers.

5. The integer execution unit performs the addition of `a` and `b`.

6. The CPU may access L1/L2 caches to fetch operands and store results, minimizing latency compared to accessing main memory.

7. The result of the addition (`c`) is stored back in the memory.

8. Instructions are retired in order, ensuring the program's logical flow is maintained.

9. The result (`8`) is sent to the output device (console) via system calls, which involve interaction with the operating system.

The CPU performs a machine cycle, which includes accessing data, performing operations, and storing results back in memory. Modern CPUs can execute billions of machine cycles per second, synchronized by the clock generator. The speed of the clock is measured in GHz (Gigahertz), where 1 GHz equals 1 billion cycles per second.

<div align="center">

![Machine Cycle](./img/machine_cycle.png)

</div>

---

Other Core CS Concepts are covered in the following sections:

- [Theoretical Computer Science](./Theoretical_Computer_Science.md)

- [Operating Systems](./Operating_System.md)

- [Computer Networks](./Computer_Networks.md)

- [Linux & Git](./Linux_GIT.md)

- [Data structures and algorithms](../03.%20Data%20Structures%20and%20Algorithms/Readme.md)

- [Databases](../04.%20Database%20Systems/Readme.md)
