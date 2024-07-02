<h1 align="center" > Operating System </h1>

An Operating system is a software that manages and handle the hardware and software resources of a computer system and provides an environment in which the user can execute programs in a convenient and efficient manner by hiding underlying complexities of hardware and acting as a resource manager.

![Overview of OS](./img/OS_overview.png)

### Functions of Operating System:

1. **Process Management**: It is a function of the operating system that handles the creation, deletion, and scheduling of `processes`.

2. **Memory Management**: OS handles the allocation and de-allocation of memory space.

3. **File Management**: OS handles the creation, deletion, and modification of files.

4. **Security & Isolation**: OS handles the security of the computer system and provides isolation between processes.

5. **User Interface**: OS acts as the interface between the user and the computer system.

### Types of Operating System:

1. **Single process OS**: Does one process at a time. 

> EG. MS-DOS

![Single process OS](./img/single_process_os.png)

2. **Batch processing OS**: Multiple processes at a time. 

> EG. UNIX

![Batch processing OS diagram](./img/Batch_OS.png)

3. **Multi programming OS**: Multiple processes at a time. It uses `context switching` to switch between processes, where we maintain a `ready queue` of processes and CPU switches between them. Before switching, CPU saves the state of the current process in `PCB` (Process Control Block). Multi-programming OS has better CPU utilization, less process starvation and supports high priority processes. _It has only a single CPU_.

>EG. Windows. 

![multiprogramming Operating system](./img/multi_programming_os.png)

4. **Multi-tasking OS**: An extension of multi-programming OS. Along with context switching, it also uses `time sharing`. In time sharing, CPU divides its time among multiple processes and thus improving the overall performance of the system. It has better response time and supports interactive processes.  

![Multi-tasking OS](./img/multi_tasking_os.png)

5. **Multi-processing OS**: In multiprocessing OS, there are multiple CPUs. Each CPU has its own memory and I/O channels.

6. **Distributed OS**: Multiple CPUs are connected to each other via a network. Each CPU has its own memory and I/O channels. Each CPU executes its own process.

![Distributed Operating System](./img/Distributed_OS.png)

7. **Real-time OS**: It is used in real-time systems where the response time is very critical. It is used in systems like flight control systems, medical systems, etc.

## Process vs Thread

### Process

- A program in execution.

### Thread

- A lightweight process. It does independent work within a process. Threads within a process share the same memory space and resources.

EG: An app that takes user input and while we fill in the data and continue to use the application, one thread may be responsible for storing that input in some cloud storage, or making some API call, etc.


## Common Questions on Operating System:

1. Difference between Multitasking and Multithreading ?

> Multitasking is the ability to execute more than one task at the same time. It would have multiple processes running at the same time. It does support isolation and memory protection between processes.
>
> Multithreading is the ability to execute more than one thread at the same time. It would only have one process running at the same time but that process would have multiple threads running at the same time. It does not support isolation and memory protection between threads.