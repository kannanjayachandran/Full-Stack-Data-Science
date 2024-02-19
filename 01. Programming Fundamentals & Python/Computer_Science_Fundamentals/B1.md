# Computer Science Concepts

This section serves as a general introduction to computer science concepts. It is not exhaustive and is meant to be a starting point for further sections.

## 1. Information representation

- Information in computers is stored as 0s and 1s (bits). Think of them as symbols, not just numbers. Depending on the application, they can represent numbers, letters, pictures, or sounds.

- The smallest unit of information is a bit. A group of 8 bits is called a byte. A byte can represent 256 different values (2^8). A byte can also represent a single character in the ASCII character set. 

- Bits represented as 0s and 1s, signify false and true values. Boolean operations are used to manipulate bits, which are the building blocks of digital information.

- A gate is a device that generates the output of a Boolean operation based on its input values. In modern computers, gates are typically made as small electronic circuits where 0s and 1s are represented by different voltage levels. They serve as the fundamental components upon which computers are built.

- The most common gates are AND, OR, and NOT. These gates can be combined to create more complex operations. For example, an XOR gate outputs true if the number of true inputs is odd.

## 2. Storage

- Computers store information in memory. Memory is divided into small units called cells. typically each holding 8 bits, known as a byte. 

- While there's no physical left or right orientation within a computer, we often visualize memory cells as linear, with the high-order end on the left and the low-order end on the right. The high-order bit, or most significant bit, holds significant value in numerical interpretation, while the low-order bit, or least significant bit, holds less weight.

- Main memory, often termed random access memory (RAM), allows independent access to cells, unlike mass storage systems (hard disk, SSD, etc) that handle data in large blocks. Modern RAM technologies, such as Dynamic RAM (DRAM) or Synchronous DRAM (SDRAM), utilize techniques like refreshing to maintain data integrity.

![Memory Hierarchy](image.png)