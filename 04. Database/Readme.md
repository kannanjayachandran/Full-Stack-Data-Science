# Databases

Databases are primary tools in software engineering. We use database to organize data so that we can query, search, manipulate and process data in easy and efficient way. Think of them like a _data model_ which cleanly abstract the way in which information is stored on the memory.

## Database Management Systems

A database management system (DBMS) is a software package designed to define, manipulate, retrieve and manage data in a database. A DBMS generally manipulates the data itself, the data format, field names, record structure and file structure. It also defines rules to validate and manipulate this data. A DBMS relieves users of framing programs for data maintenance. Fourth-generation query languages, such as SQL, are used along with the DBMS package to interact with a database. The most popular DBMSes are MySQL, PostgreSQL, SQLite, MongoDB, Redis, etc.

## Database Normalization

Database normalization is the process of structuring a relational database in accordance with a series of so-called normal forms in order to _reduce data redundancy_ and improve data integrity. It was first proposed by Edgar F. Codd as an integral part of his relational model.

Normalization entails organizing the columns (attributes) and tables (relations) of a database to ensure that their dependencies are properly enforced by database integrity constraints. It is accomplished by applying some formal rules either by a process of _synthesis_ (creating a new database design) or _decomposition_ (improving an existing database design). The most popular normal forms are 1NF, 2NF, 3NF, BCNF, 4NF, 5NF, 6NF, etc.

## Types of databases

There are different types of databases available for various use cases. The most popular ones are **relational databases**, **document databases**, **key-value databases**, **graph databases**, etc. We should choose database based on the performance and scalability requirements of our application.

### Relational Databases

Relational Data model which typically uses the Structured Query Language (SQL) to query and manipulate data, is perhaps the best known data model. Relational databases are based on them. Data is organized into relations (`known as tables in SQL`), where each relation is an unordered collection of tuples (`rows in SQL`). It was proposed by `Edgar Codd` in 1970 and by mid-80's `relational database management systems` (RDBMSes) and SQL becomes widely used.

Even now relational databases are the most popular databases. They are used in almost all the applications. They are extremely powerful while working with structured data. The most popular relational databases are MySQL, PostgreSQL, SQLite, etc.

### [SQL](./1_SQL/Basics.md)

### Document Databases

Document databases are a type of non-relational database that is designed to store and query data as JSON-like documents. Document databases make it easier for developers to store and query data in a database by using the same document-model format they use in their application code. They are used in applications where the data is not structured and the schema is not fixed. The most popular document databases are MongoDB, CouchDB, Firebase Firestore ,etc.

### [MongoDB](./2_MongoDB/Basics.md)

### Key-Value Databases

Key-value databases are a type of non-relational database that uses a simple key-value method to store data. It is a simple data model that stores data as a collection of key-value pairs. Each key is associated with only one value in a collection. The value can be a string, a number, a list, or even a complex data structure. The most popular key-value databases are Redis, DynamoDB, Apache Cassandra etc.

## Graph Databases

Graph databases are a type of non-relational database that uses graph theory to store, map and query relationships. Graph databases are basically used to store data whose relations are more important than the data itself. The most popular graph databases are Neo4j, OrientDB, etc.

## Why use database over file system ?

- **Data integrity** : Databases are designed to ensure that data is always in a consistent state and that transactions are ACID compliant {Atomicity, Consistency, Isolation, Durability}. This is not the case with file systems.

- **Data sharing** : Databases are designed to allow multiple users to access and manipulate data at the same time.

- **Data security** : Databases are designed to allow multiple users to access and manipulate data at the same time.
