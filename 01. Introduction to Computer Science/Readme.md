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

> Boolean logic forms the basis of all digital circuits and programming decisions.

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

### Related Topics

- [Theoretical Computer Science](./Theoretical_Computer_Science.md)

- [Operating Systems](./Operating_System.md)

- [Computer Networks](./Computer_Networks.md)

- [Linux & Git](./Linux_GIT.md)

- [Data structures and algorithms](../03.%20Data%20Structures%20and%20Algorithms/Readme.md)

- [Databases](../04.%20Database%20Systems/Readme.md)

---
