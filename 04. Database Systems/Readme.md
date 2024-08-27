# Database Systems

## Table of Contents

1. [Introduction to Databases](#Introduction-to-Databases)
2. [Types of Databases](#Types-of-Databases)
    - [Relational Databases](#Relational-Databases)
    - [Non-Relational Databases](#Non-Relational-Databases)
        - [Document Databases](#Document-Databases)
        - [Key-Value Databases](#Key-Value-Databases)
        - [Graph Databases](#Graph-Databases)
        - [Vector Databases](#Vector-Databases)
3. [Database Languages](#Database-Languages)
    - [Structured Query Language (SQL)](#Structured-Query-Language-SQL)
4. [Further Reading](#Further-Reading)

---

## Introduction to Databases

A **database** is an organized collection of data that is generally stored and accessed electronically from a computer system. A **Database Management System (DBMS)** is software that enables users to store, query, manipulate, and process large amounts of data in an efficient manner. Essentially, a database can be viewed as a *data model* that abstracts the way in which information is stored on disk, allowing users to interact with data without needing to understand the complexities of the underlying storage mechanisms.

## Types of Databases

Databases are categorized into different types based on their data models and the use cases they address. The two most prominent categories are **Relational Databases** and **Non-Relational Databases**.

### Relational Databases

**Relational Databases** are built on the relational data model, a concept proposed by Edgar F. Codd in 1970. In this model, data is organized into tables (also known as relations), where each table is an unordered collection of rows (tuples). These databases are typically accessed using **SQL (Structured Query Language)** or SQL-like languages.

Key features of relational databases include:

- **Structured Data Organization:** Data is organized in well-defined tables with rows and columns.

- **ACID Compliance:** Ensures **Atomicity**, **Consistency**, **Isolation**, and **Durability** of transactions.

- **Strong Data Integrity and Consistency:** Enforces data accuracy and reliability.

- **Powerful Querying Capabilities:** Supports complex queries, joins, and data manipulations.

- **Support for Transactions and CRUD Operations:** Allows for robust data operations.

- **Fixed Schema:** Data structure is predefined and remains consistent.

Popular relational databases include `MySQL`, `PostgreSQL`, `SQLite`, and `Oracle Database`.

### Non-Relational Databases

**Non-Relational Databases** do not follow the relational model; instead, they use other data models like `key-value`, `document`, and `graph`. These databases are also known as **NoSQL Databases** and are preferred when the data is unstructured or when the schema is not fixed.

#### Document Databases

**Document Databases** store and query data as JSON-like documents. They are particularly suited for applications where the document model used in application code closely matches the data storage format.

Popular document databases include `MongoDB`, `CouchDB`, `Firebase`, and `Firestore`.

#### Key-Value Databases

**Key-Value Databases** store data in simple key-value pairs, functioning like hash tables. The values can range from simple strings to complex data structures. These databases are optimized for fast reads, although write operations can be slower due to their distributed nature.

Popular key-value databases include `Redis`, `DynamoDB`, and `Apache Cassandra`.

#### Graph Databases

**Graph Databases** use graph theory to store, map, and query relationships. They are ideal for scenarios where the relationships between data points are as important as the data itself.

Popular graph databases include `Neo4j` and `OrientDB`.

#### Vector Databases

**Vector Databases** are specialized systems designed to store, manage, and query high-dimensional vector data. These databases are crucial in contexts like **Large Language Models (LLMs)**, where they efficiently store embeddings of words, sentences, or documents and optimize for similarity searches.

Popular vector databases include `Faiss` and `Pinecone`.

## Database Languages

Similar to how programming languages are used to create software, database languages are used to interact with and manipulate databases. The most widely used database language is **SQL (Structured Query Language)**, which is used primarily with relational databases.

### Structured Query Language (SQL)

**SQL** is the standard language for interacting with relational databases. It allows users to perform a variety of operations, including querying data, updating records, and managing database structures.

## Further Reading

- [Read more about SQL](./1_SQL/Readme.md)
- [Read more about MongoDB and Redis](./2_NoSQL_DB/Readme.md)
