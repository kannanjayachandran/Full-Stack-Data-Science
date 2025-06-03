<h1 align="center"> Computer Science Fundamentals </h1>

Computers were built to solve arithmetic problems, but they have since evolved to run the internet, play video games, create artificial brains, and simulate the universe. At the core, all computations boil down to manipulating `0`s and `1`s.

**[Computer science](https://en.wikipedia.org/wiki/Computer_science)** is the study of computers and computational systems. It is a broad field which includes everything from the _algorithms_ that make up _software_ to how software interacts with _hardware_.

## [Boolean Logic](https://en.wikipedia.org/wiki/Boolean_algebra)

Boolean logic is a form of algebra in which all values are either `true` or `false`. It is the basis for computer logic and programming. The fundamental operations in boolean logic are:

- **AND**: True if both operands are true.
- **OR**: True if at least one operand is true.
- **NOT**: Inverts the value of the operand (true becomes false, and vice versa).
- **XOR**: True if exactly one operand is true.
- **NAND**: True if at least one operand is false (NOT AND).
- **NOR**: True if both operands are false (NOT OR).
- **XNOR**: True if both operands are the same (NOT XOR).

## [The Computer Architecture](https://en.wikipedia.org/wiki/Computer_architecture)

The basic components of a computer are the `CPU`, `memory`, and `I/O devices`.

1. **Central Processing Unit (CPU)**: Execute instructions and performing calculations.

2. **Memory**: Stores data.
    - **Primary storage** (RAM): Stores data and instructions that the CPU needs immediately, ensuring fast access.

    - **Secondary storage**: Includes devices like hard drives (HDDs) and solid-state drives (SSDs) for long-term data storage.

3. **Input/Output (I/O) devices**: Allow the computer to interact with the outside world, enabling input and output operations. Examples include keyboards, mice, monitors, and network cards.

<div align="center">

![Computer Architecture](./img/Arch.png)

</div>

### [CPU (Central Processing Unit)](https://en.wikipedia.org/wiki/Central_processing_unit)

CPU is a complex [integrated circuit](https://en.wikipedia.org/wiki/Integrated_circuit) (IC) or die that contains multiple components working together to execute instructions and process data. The CPU contains 3 important components:

1. **Control Unit (CU)**: Directs the operation of the processor and coordinates the activities of all other components.

2. **Arithmetic Logic Unit (ALU)**: Performs arithmetic and logical operations on data.

    - **Arithmetic Operations**: Addition, subtraction, multiplication, and division.

    - **Logical Operations**: AND, OR, NOT, XOR, etc.

3. **Memory Unit or Registers**: Temporary storage locations within the CPU that hold data and instructions during processing.

CPU also contains a _graphics processor_, _floating-point units_ (FPUs), _cache memory_, and several other critical subsystems.

**[CPU Cores](https://en.wikipedia.org/wiki/Central_processing_unit)**: Modern CPUs consist of multiple cores, each capable of independently executing instructions. This parallelism enables higher performance in multitasking and computational workloads. Additionally, many CPUs support simultaneous multithreading (SMT) or hyper-threading (Intel-specific), allowing each core to handle multiple threads concurrently. 

- **Key Components of a CPU Core**

    - **Floating Point Unit** (FPU): Handles floating-point arithmetic, essential for scientific calculations, graphics processing, and other computationally intensive tasks.

    - **Integer Execution Units**: Perform basic arithmetic and logical operations on integers, crucial for general-purpose computing.
    
    - **Out of order scheduler and Retirement unit**: This scheduler allows instructions to execute out of their program order, maximizing resource utilization.
    Once the instruction are executed out of order; the retirement unit ensures instructions are completed in the correct order to maintain program integrity.
    
    - **Decode Unit**: Converts high-level instructions fetched from memory into low-level micro-operations that can be executed by the CPU.

    - **Registers**: Small, fast storage locations within the CPU used to store data temporarily during execution. 
        - **Integer Registers**: Store integer data and memory addresses.
        - **Floating Point** Registers: Store floating-point data for arithmetic operations.

    - **Cache Memory**: Multi-level cache systems are used to speed up data access for the CPU.

        - **L1 Cache**: Smallest and fastest, located directly on the core.

        - **L2 Cache**: Larger and slightly slower, shared among a subset of cores.

        - **L3 Cache**: Largest and slowest, shared across all cores in the CPU. 

**Additional Subsystems in a CPU**

- **Memory Controller**: Manages data flow between the CPU and RAM, ensuring efficient access to memory.

- **Graphics Processor**: Integrated graphics processors (iGPUs) handle basic graphical tasks, reducing the need for a dedicated GPU in some systems.

- **Interconnects**: Facilitate communication between cores, cache, memory, and other components.

The following is a labelled die shot of `13-th gen Intel Core i9 processor` (Raptor Lake) with 24 cores.

![Die shot of a chip](./img/die_shot.png)

> Image Courtesy : [Wikipedia](https://upload.wikimedia.org/wikipedia/commons/a/a4/Intel_Core_i9-13900K_Labelled_Die_Shot.jpg)

- > The above `13-th gen Intel i9` CPU has 24 cores and 48 threads.

- > Inside every CPU core we can find a layout of around 44,000 transistors physically executing 32-bit instructions, with a grand total of around 26 Million! transistors in the entire processor.

### The Machine Cycle

The CPU operates through repeated machine cycles, which consist of:

- **Fetch**: Retrieving instructions or data.

- **Decode**: Translating instructions into executable operations.

- **Execute**: Performing the required operation.

- **Store**: Writing results back to memory.

<div align="center">

![Machine Cycle](./img/machine_cycle.png)

</div>


### Memory

> How we store data.

To efficiently manage data access and processing, computer systems employ a hierarchy of memory. Each level of the hierarchy balances **speed**, **capacity**, and **cost**:

<p align="center"><img src="./img/Memory_hierarchy.png" alt="Machine Cycle" height=390></p>

- **Registers**: The fastest and smallest memory units, located within the CPU. Registers store data temporarily

- **CPU Cache**: Extremely fast but small and expensive memory located close to the CPU.

- **RAM** (Main Memory): Moderately fast and larger than cache, but more affordable.

- **Secondary Storage**: Large-capacity, slower memory such as Hard Disk Drives (HDDs) and Solid-State Drives (SSDs).

**Memory Organization**

Computer memory is divided into cells, with each cell typically holding 1 byte of data. While memory has no physical orientation, it is often visualized linearly for clarity:

![Computer Memory](./img/computer_memory_diagram.png)

- **High-Order End**: The leftmost side, containing the most significant bit (`MSB`).

- **Low-Order End**: The rightmost side, containing the least significant bit (`LSB`).

For example, in a byte represented as `10110011`:

- High-Order Bit: 1 (leftmost).

- Low-Order Bit: 1 (rightmost).

**Information Representation in Computers**

Computers represent all information as `binary data`, a sequence of **0s** and **1s** called **bits**.

- **1 bit**: The smallest unit of data in a computer, representing two possible states (0 or 1).

- **8 bits**: Equal to _1 byte_, the fundamental unit of storage in most computer systems.

A single byte can represent up to **256 combinations** ($2^8$), making it capable of storing numbers, characters, and other basic data types.

<details><summary>

### Python Program Execution</summary>

We will explore [Python](https://en.wikipedia.org/wiki/Python_(programming_language)) language in detail in upcoming sections, but for now, let's see how a simple python program gets executed in a computer.

*Consider the following python code that adds two numbers*:

```python
a = 5
b = 10

c = a + b
print(c)

# 5 + 10 = 15
```

When we run this code, the following steps occur:

- **Source Code**: The code is written in a high-level language (Python), probably using a text editor or an [Integrated Development Environment](https://en.wikipedia.org/wiki/Integrated_development_environment) (IDE).

- **Python source code execution**: We can execute the code using the Python interpreter, which converts the high-level code into machine code. In the terminal or command prompt, we can run the code using the following command `python filename.py`

- **Compilation**: Python is an [interpreted](https://en.wikipedia.org/wiki/Interpreter_(computing)) language, meaning it is executed line by line. The Python interpreter reads the code, converts it into machine code, and executes it.

    - **[Lexical Analysis](https://en.wikipedia.org/wiki/Lexical_analysis) (Tokenization)**: The Python interpreter reads the source code and breaks it into tokens (keywords, operators, and identifiers).

    - **[Parsing](https://en.wikipedia.org/wiki/Parsing) & [AST](https://en.wikipedia.org/wiki/Abstract_syntax_tree) Generation**: The interpreter checks the syntax of the code to ensure it follows the rules of the Python language. It is done by parsing the tokens to generate the _Abstract Syntax Tree (AST)_, a tree-like representation of the code structure.

    - **Bytecode Compilation**: The AST is compiled into Python bytecode, a platform-independent intermediate representation.

    - **Execution by Python Virtual Machine (PVM):** The PVM executes the bytecode instruction by instruction. Each bytecode instruction follows a fetch-decode-execute cycle:

        - **Fetch**: Retrieves the next instruction.

        - **Decode**: Translates the instruction into an operation.

        - **Execute**: Performs the operation using a stack-based virtual machine.

- **CPU Execution**:

1. **Instruction Fetch**:
The CPU fetches the bytecode instructions from memory.

1. **Instruction Decode**:
The decode unit translates the bytecode into micro-operations (`μOps`), which the CPU can execute.

1. **Out-of-Order Execution**:
The CPU executes instructions as resources become available, rather than strictly following their order in the program.

1. **Loading Data into Registers**:
The integer values (a = 5 and b = 3) are loaded into CPU registers for fast access.

1. **Performing the Addition**:
The Integer Execution Unit performs the addition (a + b) using the loaded values.

1. **Cache Access**:
The CPU may access L1/L2 caches to fetch operands or store results, reducing latency compared to accessing main memory.

1. **Storing the Result**:
The result (c = 8) is written back to memory.

1. **Instruction Retirement**:
Instructions are retired in the correct program order to maintain logical flow.

1. **Output to Console**:
The result (8) is sent to the console via system calls, which interact with the operating system to display the output.

</details>

Other Core CS Concepts are covered in the following sections:

- [Theoretical Computer Science](./Theoretical_Computer_Science.md)

- [Operating Systems](./Operating_System.md)

- [Computer Networks](./Computer_Networks.md)

- [Linux & Git](./Linux_GIT.md)

- [Data structures and algorithms](../03.%20Data%20Structures%20and%20Algorithms/Readme.md)

- [Databases](../04.%20Database%20Systems/Readme.md)
