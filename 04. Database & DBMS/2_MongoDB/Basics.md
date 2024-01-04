<h1 align="center"> Mongo DB </h1>

Mongo DB is a NOSQL, document based, multi-modal, scalable and distributed database. One of the major issue with relational database is that, they have fixed schema and it is nightmare to change the schema once the database is in production. Also fixed schema forces computationally expensive joins and group-by operations. This is where Mongo DB shines with its flexible schema. It was introduced in 2009. Although mongo db is a document database, it is slowly becoming a multi model database, by supporting features like in-memory storage, hash based indexing, etc.

![Mongo DB example](image.png)

Mongo DB stores data in the form of documents; specifically BSON (Binary JSON). BSON is a binary encoded JSON. It extends the JSON model to provide additional data types, ordered fields, and to be efficient for encoding and decoding within different languages. 

![relational db with mongo db](image-1.png)


In mongo db we can also embed one document inside another document.

## mongo syntax

- `mongosh` : Opens the mongo shell.
- `show dbs` : Shows all the databases.
- `use <db_name>` : Switches to the database.
- `db` : Shows the current database.
- `show collections` : Shows all the collections in the current database.
- `db.<collection_name>.insertOne({<document>})` : Inserts a document into the collection.
> `db.user.insertOne({name: "John Doe", age: 25})` : where db refers to the current db, user is the collection name and name and age are the fields. We have options like `insertMany` and `insert` also.
- `db.<collection_name>.find()` : Shows all the documents in the collection.
- `db.createCollection(<collection_name>)` : Creates a collection.

The `_id` field is automatically added to every document by mongo db. It is the default primary key of the document. It is a 12 byte hexadecimal number. The first 4 bytes are the timestamp, next 3 bytes are the machine id, next 2 bytes are the process id and the last 3 bytes are the counter. The counter is randomly generated. The `_id` field is indexed by default. It is immutable. We can specify our own `_id` field also.

`db.col.insertOne({some_data}, {writeConcern: { "w", "majority" j: true, wtimeout: 2000 }})` : w is the write concern, j is the journal and wtimeout is the timeout {after 2 seconds if the write operation is not successful then drop it}

`writeConcern` deals with the number of nodes that should acknowledge the write operation. `w: 1` means that only one node should acknowledge the write operation. `j: true` means that the write operation should be written to the journal. `wtimeout: 2000` means that the write operation should timeout after 2 seconds.

`find().explain("executionStats")` : Shows the execution stats of the query. This shows the number of documents scanned, number of documents returned, etc. Basically what happens under the hood when we run the query is shown here. It is highly useful for query optimization.

> pymongo is the python driver for mongo db.