# Database Management Systems (DBMS)

A **Database Management System (DBMS)** is software that enables users to interact with databases. They manage the underlying database and ensures that data is stored, retrieved, and manipulated efficiently and securely. It provides an interface for performing various operations such as creating, updating, deleting, and querying data. Common DBMSs include `MySQL`, `PostgreSQL`, `SQLite`, `MongoDB`, `Redis`, etc.

## Interaction with Databases

Generally we use **SQL (Structured Query Language)** or a similar query languages to interact with databases. SQL allows users to define, manipulate, and query data stored in a relational database.

## Advantages of Using a Database Over a File System

Using a database offers several advantages over traditional file systems, particularly in terms of data management, integrity, and security:

- **Data Integrity:** Databases ensure that data remains consistent and accurate through the use of ACID (Atomicity, Consistency, Isolation, Durability) properties. This guarantees that transactions are processed reliably, which is not inherently the case with file systems.

- **Data Sharing:** Databases are designed to support concurrent access, allowing multiple users to interact with the data simultaneously without causing conflicts or inconsistencies.

- **Data Security:** Databases provide robust security features, including access controls, encryption, and auditing, to protect sensitive information. This level of security is typically more advanced than what file systems offer.

- **Data Abstraction:** Databases abstract the complexities of data storage and management, providing users with a simplified interface to work with data. This abstraction layer allows users to focus on data manipulation without needing to understand the underlying storage details.

- **Data Independence:** Databases support data independence, meaning the structure of the data can change without affecting the application that uses the data. This flexibility is a key advantage over file systems, where changes to data structure can have significant implications.

- **Data Redundancy Reduction:** Databases are designed to minimize data redundancy, which helps prevent inconsistencies and saves storage space. In contrast, file systems often lead to duplicated data, increasing the risk of errors.

- **Data Consistency:** Databases ensure that data is consistent across the system, even in cases of multiple concurrent operations. This consistency is maintained through various mechanisms, such as transaction management and integrity constraints.

## Database Normalization

**Database Normalization** is the process of modifying the structure of a relational database to reduce data redundancy and improve data integrity. This process, first proposed by _Edgar F. Codd_ as part of his relational model, involves structuring the database's tables (relations) and columns (attributes) to ensure that their dependencies are properly enforced.

### Goals of Normalization

Normalization aims to:

- **Reduce Data Redundancy:** By eliminating duplicate data, normalization helps maintain consistency and efficiency within the database.

- **Improve Data Integrity:** Ensures that the data remains accurate and reliable, minimizing the chances of anomalies or inconsistencies.

### Normal Forms

Normalization is typically achieved through a series of stages known as **normal forms**. These forms represent different levels of database organization, each addressing specific types of redundancy and dependency issues:

1. **First Normal Form (1NF):** Ensures that the data is stored in a tabular format with no repeating groups or arrays.

2. **Second Normal Form (2NF):** Builds on 1NF by ensuring that all non-key attributes are fully dependent on the primary key.

3. **Third Normal Form (3NF):** Further refines the structure by ensuring that all attributes are not only dependent on the primary key but also independent of each other.

4. **Boyce-Codd Normal Form (BCNF):** A stronger version of 3NF, addressing specific types of anomalies not covered by 3NF.

5. **Fourth Normal Form (4NF) and beyond:** These advanced forms address more complex scenarios, such as multi-valued dependencies and join dependencies.

Normalization can be applied through a process of **synthesis** (creating a new database design) or **decomposition** (improving an existing database design). 
