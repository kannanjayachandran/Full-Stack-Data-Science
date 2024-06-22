<h1 align="center" style="color: orange"> Database And DBMS </h1>

We use database to organize data so that we can query, search, manipulate and process large amount of data in easy and efficient way. Database ensure that data is always in a consistent state and transactions are ACID compliant {Atomicity, Consistency, Isolation, Durability}.

> Think of database like a _data model_ which cleanly abstract the way in which information is stored on the memory. 

## Types of databases

There are different types of databases available for various use cases. The most popular ones are **Relational databases** and **Non-relational databases**. We should choose database based on the performance and scalability requirements of our application.

**[Read about DBMS](./DBMS.md)**

### Relational Databases

Databases built on the **relational data model**, a concept proposed by `Edgar F. Codd` in 1970. It organizes data into relations (also known as tables), where each relation is an unordered collection of tuples (rows). Relational databases typically use SQL (Structured Query Language) or SQL-like query languages to interact with and manipulate data.

Key features of relational databases include:

1. Structured data organization.

1. ACID (Atomicity, Consistency, Isolation, Durability) compliance.

1. Strong data integrity and consistency.

1. Powerful querying capabilities.

1. Support for complex joins, transactions and CRUD operations.

**CRUD** operations; 

1. Create/Insert
1. Select
1. Update
1. Delete/Drop.

- Most popular relational databases are `MySQL`, `PostgreSQL`, `SQLite`,`Oracle database`, etc.

### [SQL](./1_SQL/Basics.md)

### Non-Relational Databases

They do not use the relational data model instead they use other data models like `key-value`, `document`, `graph`, etc. They are also known as `NoSQL` databases. We use them when the data is not structured and the schema is not fixed. The most popular non-relational databases are `MongoDB`, `Redis`, `Firestore`, etc.

One of the key feature of non-relational databases is that they are horizontally scalable.

**Document Databases**

A type of non-relational database designed to store and query data as `JSON-like` documents. They makes it easier for developers to store and query data in a database by using the same _document-model_ format used in their application code. 

- Most popular document databases are `MongoDB`, `CouchDB`, `Firebase`, `Firestore` ,etc.

### [MongoDB](./2_MongoDB/Basics.md)

**Key-Value Databases**

A type of non-relational database that uses simple key-value method to store data (Hash tables). The value can be a string, a number, a list, or even a complex data structure. They can be considered as distributed hash tables. They store data in the memory itself, and heavily use techniques like sharding, horizontal scaling along with distributed computing. Their reading time is very fast, but writing time is slow.

- Most popular key-value databases are `Redis`, `DynamoDB`, `Apache Cassandra` etc.

### [Redis](./3_Reddis/Basics.md)

**Graph Databases**

A type of non-relational database that uses graph theory to store, map and query relationships. Graph databases are used to store data whose relations are more important than the data itself. 

- Most popular graph databases are `Neo4j`, `OrientDB`, etc.


## Why non-relational databases came into the picture ?

During the 70s and late 90s, disk space was the primary concern. Consequently, preference was given to disk space over latency. This paved the way for the emergence of relational databases. They were designed to store data in a structured manner and ensure data consistency.

From the early 2000s, due to the internet boom, the volume of data started to increase exponentially. The focus then shifted from disk space to latency. As a result, preference was given to latency over disk space. This marked the rise of non-relational databases.

One significant development during this time was the decreasing cost of compute power and memory. Disk space and compute power became more affordable. This period also witnessed the growing popularity of distributed computing. Google revolutionized distributed computing with their MapReduce paper, leading to the birth of Apache Hadoop. All these advancements laid the foundation for the growth of non-relational databases.

## Vector Databases 

Vector databases are specialized database systems designed to store, manage, and efficiently query high-dimensional vector data. They are important in the context of LLM (Large Language Models) as;

- They can store and query high-dimensional vectors efficiently.

- They can be used to store embeddings of words, sentences, or documents.

- Optimize for similarity searches.

- Enable fast retrieval of semantically similar information.

- Most popular vector databases are `Faiss`, `Pinecone`, etc.

### NOTE

As a data scientist or machine learning engineer, it's essential to understand how to effectively use various types of databases, including both relational and non-relational systems. This knowledge is crucial for managing and analyzing data efficiently. While a broad understanding of database types is important, you don't necessarily need to delve into the intricate details of database architecture, such as:

- Internal data structures

- Data organization methods

- Low-level implementation details

However, if you're aiming for a senior role or a position that involves strategic decision-making, a deeper understanding of database systems becomes more relevant. In such cases, you may need to:

- Evaluate different database solutions

- Make informed choices about database selection

- Optimize database performance for specific use cases

For most data science and machine learning roles, focusing on practical database usage and query optimization will suffice. This allows you to concentrate on extracting insights from data and building effective models.
