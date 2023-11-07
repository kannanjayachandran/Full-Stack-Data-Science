<h1 align="center">Redis</h1>

Redis is an open source, in memory data structure store, used as a database, cache, message broker and streaming engine. Redis provides data structures such as strings, hashes, lists, sets, sorted sets with range queries, bitmaps, hyper-log, geo-spatial indexes, and streams. Redis often serve as a data structure layer between a relational database and the application.

Redis is known for its speed and versatility. It stores data in key-value pairs and supports various data types. Some common use cases of redis are;

- **Caching** : can be used to store frequently accessed data in memory, reducing the need to fetch it from slower data sources, such as databases. This can significantly improve application performance.

- **Session management in distributed environments** : can be used to store session data in memory, which can be accessed across multiple servers in a distributed setup. This can help in building scalable applications.

- **Pub/Sub messaging** : can be used to implement a messaging system where publishers and subscribers can send and receive messages. This can be used to implement chat applications or notification systems, for example.

- **Real time analytics** : can be used to store data in real time, which can be used for analytics. This can be used to implement leader-boards, IOT-device data, for example.

## Working with Redis

Redis has clients for almost all major programming languages. After installing Redis, you can start the Redis server by running the following command in the terminal.

```bash
redis-server
```

You can then start the Redis client by running the following command in another terminal.

```bash
redis-cli
```

You can then start executing Redis commands in the Redis client. For example, you can set a key-value pair using the `SET` command. Although case doesn't matter, it is better to use uppercase for commands and lowercase for keys and values.

```bash
SET name "John Doe"

GET name
```

You can also use the `EXISTS` command to check if a key exists.

```bash
EXISTS name
```

You can use the `DEL` command to delete a key and use the `FLUSHALL` command to delete all keys.

```bash
DEL name
FLUSHALL
```

You can use the `KEYS` command to list all keys.

```bash
KEYS *
```
We can set a key with an expiry time using the `EXPIRE` command.

```bash
SET name "John Doe"
EXPIRE name 10
```
> Note: The expiry time is in seconds.

You can use the `TTL` command to check the time to live of a key.

```bash
TTL name
```
> Note: The time to live is in seconds.

You can use the `PERSIST` command to remove the expiry time of a key.

```bash
PERSIST name
```

we can set a key value and expiration in one command using the `SETEX` command.

```bash
SETEX name 10 "John Doe"
```
