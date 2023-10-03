<h1 align="center"> Databases </h1>

Databases are primary tools in software engineering. We use database to organize data so that we can query, search, manipulate and process data in easy and efficient way. Think of them like a _data model_ which cleanly abstract the way in which information is stored on the memory.

## Database Management Systems

A database management system (DBMS) is a software package designed to define, manipulate, retrieve and manage data in a database. It also defines rules to validate and manipulate this data. SQL is used along with the DBMS package to interact with a database. The most popular DBMS's are MySQL, PostgreSQL, SQLite, MongoDB, Redis, Firestore, etc.

## Types of databases

There are different types of databases available for various use cases. The most popular ones are **Relational databases** and **Non-relational databases**. We should choose database based on the performance and scalability requirements of our application.

### Relational Databases

Relational Data model typically uses SQL to query and manipulate data. Relational databases are based on this relational data model. Data is organized into relations (`known as tables in SQL`), where each relation is an unordered collection of tuples (`rows in SQL`). It was proposed by `Edgar Codd` in 1970 and by mid-80's `relational database management systems` (RDBMS's) and SQL becomes widely used.

Even now relational databases are the most popular databases. They are used in almost all the applications. They are extremely powerful while working with structured data. The most popular relational databases are MySQL, PostgreSQL, SQLite, etc.

### [SQL](./1_SQL/Basics.md)

### Non-Relational Databases

Non-relational databases are the databases that don't use the relational data model. They are also known as NoSQL databases. They are used in applications where the data is not structured and the schema is not fixed. The most popular non-relational databases are MongoDB, Redis, Firestore, etc.

**Document Databases**

Document databases are a type of non-relational database that is designed to store and query data as JSON-like documents. They makes it easier for developers to store and query data in a database by using the same document-model format they use in their application code. They are used in applications where the data is not structured and the schema is not fixed. The most popular document databases are MongoDB, CouchDB, Firebase Firestore ,etc.

### [MongoDB](./2_MongoDB/Basics.md)

**Key-Value Databases**

Key-value databases are a type of non-relational database that uses a simple key-value method to store data (Hash tables). It is a simple data model that stores data as a collection of key-value pairs. Each key is associated with only one value in a collection. The value can be a string, a number, a list, or even a complex data structure. We can say that they are simply distributed hash tables. They store data in the memory itself, and heavily use techniques like sharding, horizontal scaling along with distributed computing. Their reading time is very fast, but writing time is a little bit slow.

The most popular key-value databases are Redis, DynamoDB, Apache Cassandra etc.

### [Redis](./3_Reddis/Basics.md)

**Graph Databases**

Graph databases are a type of non-relational database that uses graph theory to store, map and query relationships. Graph databases are basically used to store data whose relations are more important than the data itself. The most popular graph databases are Neo4j, OrientDB, etc.


## Why non-relational databases came into the picture ?

During the 70's and late 90's the biggest concern was disk space. It was an absolute necessary to avoid data redundancy at any cost. Hence preference were given to disk space over latency. Also the volume of data was not that big. It was "OK" to join tables.

From the early 2000's because of the internet boom, the volume of data started to increase exponentially. Now the biggest concern was latency. Hence preference were given to latency over disk space. This is where non-relational databases came into the picture. One significant advent of this time was compute power and memory became cheaper. Disk space was not that expensive anymore. This was also the period when distributed computing become popular. Google revolutionized the distributed computing with their MapReduce paper and following that Apache Hadoop was born. All these paved the way for non-relational databases to rise.

## Why use database over file system ?

- **Data integrity** : Databases are designed to ensure that data is always in a consistent state and that transactions are ACID compliant {Atomicity, Consistency, Isolation, Durability}. This is not the case with file systems.

- **Data sharing** : Databases are designed to allow multiple users to access and manipulate data at the same time.

- **Data security** : Databases are designed to allow multiple users to access and manipulate data at the same time.

## Database Normalization

Database normalization is the process of structuring a relational database in accordance with a series of so-called normal forms in order to _reduce data redundancy_ and improve data integrity. It was first proposed by Edgar F. Codd as an integral part of his relational model.

Normalization entails organizing the columns (attributes) and tables (relations) of a database to ensure that their dependencies are properly enforced by database integrity constraints. It is accomplished by applying some formal rules either by a process of _synthesis_ (creating a new database design) or _decomposition_ (improving an existing database design). The most popular normal forms are 1NF, 2NF, 3NF, BCNF, 4NF, 5NF, 6NF, etc.






> <picture>
>   <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Mqxx/GitHub-Markdown/main/blockquotes/badge/light-theme/note.svg">
>   <img alt="Note" src="https://raw.githubusercontent.com/Mqxx/GitHub-Markdown/main/blockquotes/badge/dark-theme/note.svg">
> </picture><br>
>
> For a data scientist or machine learning engineer, it's crucial to know how to use different types of databases how to work with both relational and non-relational databases, But you don't have to get into the nitty-gritty details of how a database is built, what kinds of data structures it uses, or how it organizes data internally, unless you're aiming for a senior role where you will be involved in the decision making of choosing a database or something like that.
