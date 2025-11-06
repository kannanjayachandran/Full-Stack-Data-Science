<h1 align="center"> NoSQL Databases - Mongo DB - Redis DB </h1>

## Table of Contents

1. [Why NoSQL?](#why-nosql)
2. [MongoDB](#mongodb)
3. [Redis](#redis)
4. [NoSQL for ML Workflows](#nosql-for-ml-workflows)

---

## The Shift from RDBMS to NoSQL

| Era | Primary Concern | Database Type | Driver |
|-----|-----------------|---------------|--------|
| **1970s-1990s** | Disk space expensive | Relational (SQL) | Data normalization to save space |
| **2000s-Present** | Latency critical | Non-relational (NoSQL) | Internet scale, distributed computing |

**Key catalysts for NoSQL adoption:**
- Exponential data growth (social media, IoT, web logs)
- Decreasing storage costs (disk space cheap)
- Need for horizontal scalability (scale out vs. scale up)
- Google's MapReduce paper → Apache Hadoop ecosystem
- Rise of cloud computing and distributed systems

## When to Use NoSQL vs SQL

| Use NoSQL When | Use SQL When |
|----------------|--------------|
| Flexible/evolving schema | Fixed, well-defined schema |
| Horizontal scaling required | Vertical scaling sufficient |
| High write throughput | Complex joins and transactions |
| Unstructured/semi-structured data | Strong consistency required |
| Low-latency reads (caching) | ACID guarantees essential |

---

## MongoDB

**MongoDB** is a document-oriented NoSQL database that stores data in BSON (Binary JSON) format.

**Key characteristics:**
- Schema-flexible (documents in same collection can have different fields)
- Distributed and horizontally scalable (sharding)
- Rich query language with indexes
- No expensive JOINs (embed related data in documents)
- Ideal for rapidly evolving data models

### RDBMS vs MongoDB Mapping

| RDBMS | MongoDB |
|-------|---------|
| Database | Database |
| Table | Collection |
| Row | Document |
| Column | Field |
| JOIN | Embedded documents or `$lookup` |
| Primary Key | `_id` field (auto-generated) |

---

### Basic MongoDB Operations

#### Connection and Database Management

```js
// Start MongoDB shell
mongosh

// List all databases
show dbs

// Switch to database (creates if doesn't exist)
use analytics_db

// Check current database
db

// List collections
show collections

// Create collection explicitly
db.createCollection('users')
```

---

#### CRUD Operations

**Create (Insert)**

```javascript
// Insert single document
db.users.insertOne({
    name: "Alice",
    age: 28,
    email: "alice@example.com",
    skills: ["Python", "SQL", "ML"],
    created_at: new Date()
})

// Insert multiple documents
db.users.insertMany([
    { name: "Bob", age: 32, city: "NYC" },
    { name: "Charlie", age: 25, city: "SF" }
])

// _id is auto-generated if not provided
// Output: { acknowledged: true, insertedId: ObjectId("...") }
```

**Read (Query)**

```javascript
// Find all documents
db.users.find()

// Find with filter
db.users.find({ age: { $gte: 25 } })  // age >= 25

// Find one document
db.users.findOne({ name: "Alice" })

// Projection (select specific fields)
db.users.find({ age: { $gte: 25 } }, { name: 1, age: 1, _id: 0 })

// Sort and limit
db.users.find().sort({ age: -1 }).limit(5)  // Top 5 oldest users

// Count documents
db.users.countDocuments({ city: "NYC" })
```

**Update**

```javascript
// Update one document
db.users.updateOne(
    { name: "Alice" },
    { $set: { age: 29, last_login: new Date() } }
)

// Update multiple documents
db.users.updateMany(
    { city: "NYC" },
    { $set: { timezone: "EST" } }
)

// Increment field
db.users.updateOne(
    { name: "Alice" },
    { $inc: { login_count: 1 } }  // Increment by 1
)

// Add to array
db.users.updateOne(
    { name: "Alice" },
    { $push: { skills: "TensorFlow" } }
)

// Upsert (insert if not exists)
db.users.updateOne(
    { name: "Diana" },
    { $set: { age: 30, city: "LA" } },
    { upsert: true }
)
```

**Delete**

```javascript
// Delete one document
db.users.deleteOne({ name: "Bob" })

// Delete multiple documents
db.users.deleteMany({ age: { $lt: 18 } })

// Delete all documents in collection
db.users.deleteMany({})
```

---

### Query Operators

```javascript
// Comparison operators
db.orders.find({ amount: { $gt: 100 } })           // Greater than
db.orders.find({ amount: { $gte: 100, $lte: 500 } }) // Between 100-500
db.orders.find({ status: { $in: ['pending', 'shipped'] } })
db.orders.find({ status: { $ne: 'cancelled' } })   // Not equal

// Logical operators
db.orders.find({
    $and: [
        { amount: { $gt: 100 } },
        { status: 'completed' }
    ]
})

db.orders.find({
    $or: [
        { amount: { $gt: 1000 } },
        { customer_type: 'VIP' }
    ]
})

// Array operators
db.users.find({ skills: { $all: ['Python', 'ML'] } })  // Has both skills
db.users.find({ skills: { $size: 3 } })  // Exactly 3 skills
db.users.find({ 'address.city': 'NYC' })  // Nested field query

// Regular expressions
db.users.find({ name: { $regex: /^A/i } })  // Name starts with 'A' (case-insensitive)

// Exists
db.users.find({ phone: { $exists: true } })  // Has phone field
```

---

### Aggregation Pipeline

MongoDB's aggregation framework is similar to SQL's GROUP BY with more flexibility.

```javascript
// Calculate total revenue by customer
db.orders.aggregate([
    { $match: { status: 'completed' } },  // WHERE clause
    { $group: {                           // GROUP BY
        _id: '$customer_id',
        total_spent: { $sum: '$amount' },
        order_count: { $sum: 1 },
        avg_order: { $avg: '$amount' }
    }},
    { $sort: { total_spent: -1 } },       // ORDER BY
    { $limit: 10 }                        // LIMIT
])

// Multi-stage pipeline
db.orders.aggregate([
    // Stage 1: Filter recent orders
    { $match: { 
        order_date: { $gte: new Date('2024-01-01') }
    }},
    
    // Stage 2: Join with products (like SQL JOIN)
    { $lookup: {
        from: 'products',
        localField: 'product_id',
        foreignField: '_id',
        as: 'product_info'
    }},
    
    // Stage 3: Unwind array
    { $unwind: '$product_info' },
    
    // Stage 4: Project fields
    { $project: {
        customer_id: 1,
        product_name: '$product_info.name',
        amount: 1,
        month: { $month: '$order_date' }
    }},
    
    // Stage 5: Group and aggregate
    { $group: {
        _id: { customer: '$customer_id', month: '$month' },
        monthly_spend: { $sum: '$amount' },
        products_bought: { $addToSet: '$product_name' }
    }}
])
```

**Common aggregation operators:**
- `$sum`, `$avg`, `$min`, `$max`, `$stdDevPop`
- `$first`, `$last`, `$push`, `$addToSet`
- `$match`, `$group`, `$sort`, `$limit`, `$skip`
- `$project`, `$lookup` (JOIN), `$unwind`

---

### Indexes and Performance

```javascript
// Create index on single field
db.users.createIndex({ email: 1 })  // 1 for ascending, -1 for descending

// Compound index
db.orders.createIndex({ customer_id: 1, order_date: -1 })

// Text index for full-text search
db.products.createIndex({ description: 'text' })
db.products.find({ $text: { $search: 'machine learning' } })

// List indexes
db.users.getIndexes()

// Drop index
db.users.dropIndex('email_1')

// Query performance analysis
db.orders.find({ customer_id: 123 }).explain('executionStats')
/*
Look for:
- executionTimeMillis: Query execution time
- totalDocsExamined: Documents scanned
- totalKeysExamined: Index keys scanned
- executionStages.stage: "IXSCAN" (index scan) is good, "COLLSCAN" (collection scan) is slow
*/
```

**Performance tips:**
- ✅ Create indexes on frequently queried fields
- ✅ Use compound indexes for multi-field queries
- ❌ Avoid too many indexes (slows writes)
- ✅ Use projection to return only needed fields
- ✅ Limit embedded documents depth (avoid deeply nested structures)

---

### MongoDB with Python (PyMongo)

```python
from pymongo import MongoClient
from datetime import datetime
import pandas as pd

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['analytics_db']
collection = db['orders']

# Insert documents
orders = [
    {'customer_id': 'C001', 'amount': 150.50, 'status': 'completed', 'date': datetime(2024, 1, 15)},
    {'customer_id': 'C002', 'amount': 200.00, 'status': 'pending', 'date': datetime(2024, 1, 16)}
]
collection.insert_many(orders)

# Query documents
results = collection.find({'amount': {'$gt': 100}})
for doc in results:
    print(doc)

# Query with projection
results = collection.find(
    {'status': 'completed'},
    {'customer_id': 1, 'amount': 1, '_id': 0}
)

# Convert to pandas DataFrame
df = pd.DataFrame(list(results))
print(df)

# Aggregation pipeline
pipeline = [
    {'$match': {'status': 'completed'}},
    {'$group': {
        '_id': '$customer_id',
        'total_spent': {'$sum': '$amount'},
        'order_count': {'$sum': 1}
    }},
    {'$sort': {'total_spent': -1}}
]

agg_results = collection.aggregate(pipeline)
df_agg = pd.DataFrame(list(agg_results))

# Update documents
collection.update_many(
    {'status': 'pending'},
    {'$set': {'status': 'processing'}}
)

# Delete documents
collection.delete_many({'amount': {'$lt': 10}})

# Close connection
client.close()
```

---

### Write Concerns and Durability

```javascript
// Write concern levels
db.users.insertOne(
    { name: "Alice", email: "alice@example.com" },
    { 
        writeConcern: {
            w: 'majority',  // Acknowledged by majority of nodes
            j: true,        // Written to journal (durable)
            wtimeout: 5000  // Timeout after 5 seconds
        }
    }
)

/*
w options:
- 1: Acknowledged by primary (default)
- 'majority': Acknowledged by majority of replica set
- 0: No acknowledgment (fastest, least safe)

j: true ensures data is written to journal (survives crashes)
*/
```

### MongoDB for ML

**Use cases:**
- Store raw training data (flexible schema for varying features)
- Log experiment metadata (hyperparameters, metrics)
- Store model artifacts metadata
- Feature store (document per entity with nested features)

```python
from pymongo import MongoClient
import pandas as pd

client = MongoClient('mongodb://localhost:27017/')
db = client['ml_platform']

# Store experiment results
experiments = db['experiments']
experiments.insert_one({
    'experiment_id': 'exp_001',
    'model_type': 'xgboost',
    'hyperparameters': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100
    },
    'metrics': {
        'accuracy': 0.92,
        'precision': 0.89,
        'recall': 0.91
    },
    'features_used': ['age', 'income', 'tenure'],
    'created_at': datetime.now()
})

# Query best models
best_models = experiments.find(
    {'metrics.accuracy': {'$gt': 0.90}}
).sort('metrics.accuracy', -1).limit(5)

# Export training data to pandas
training_data = db['customer_features'].find({'label': {'$exists': True}})
df = pd.DataFrame(list(training_data))
```


---

## Redis

**Redis** (Remote Dictionary Server) is an in-memory key-value store known for:
- Sub-millisecond latency
- Rich data structures (strings, hashes, lists, sets, sorted sets)
- Atomic operations
- Built-in replication and persistence
- Pub/Sub messaging

**Common use cases:**
- **Caching**: Store frequently accessed data
- **Session management**: Distributed session store
- **Real-time analytics**: Leaderboards, counters
- **Message queues**: Task queues, pub/sub
- **Rate limiting**: API throttling

---

### Redis Data Types

| Type | Description | Use Case |
|------|-------------|----------|
| **String** | Binary-safe strings (up to 512MB) | Cache values, counters |
| **Hash** | Key-value pairs within a key | Store objects (user profiles) |
| **List** | Ordered collection, doubly linked | Queues, timelines |
| **Set** | Unordered unique strings | Tags, unique visitors |
| **Sorted Set** | Set with scores | Leaderboards, time-series |
| **Bitmap** | Bit arrays | User activity tracking |
| **HyperLogLog** | Probabilistic cardinality estimator | Unique counts |

---

### Basic Redis Commands

```bash
# Start Redis server
redis-server

# Start Redis client
redis-cli

# ===== STRING OPERATIONS =====
SET user:1:name "Alice"
GET user:1:name  # Returns: "Alice"

# Set with expiry (in seconds)
SETEX session:abc123 3600 "user_data"  # Expires in 1 hour

# Set if not exists
SETNX user:1:email "alice@example.com"

# Increment/Decrement (atomic)
SET page_views 100
INCR page_views        # 101
INCRBY page_views 5    # 106
DECR page_views        # 105

# Multiple set/get
MSET key1 "value1" key2 "value2"
MGET key1 key2

# ===== KEY MANAGEMENT =====
EXISTS user:1:name     # Returns 1 (exists) or 0 (doesn't exist)
DEL user:1:name        # Delete key
KEYS user:*            # List all keys matching pattern (avoid in production)
EXPIRE user:1:name 60  # Set expiry to 60 seconds
TTL user:1:name        # Check time to live
PERSIST user:1:name    # Remove expiry
FLUSHALL               # Delete all keys (⚠️ use carefully)

# ===== HASH OPERATIONS =====
HSET user:1 name "Alice" age 28 email "alice@example.com"
HGET user:1 name                 # Returns: "Alice"
HGETALL user:1                   # Returns all fields
HINCRBY user:1 age 1             # Increment age
HDEL user:1 email                # Delete field

# ===== LIST OPERATIONS =====
LPUSH tasks "task1" "task2"      # Push to left (head)
RPUSH tasks "task3"              # Push to right (tail)
LPOP tasks                       # Pop from left
RPOP tasks                       # Pop from right
LRANGE tasks 0 -1                # Get all elements
LLEN tasks                       # List length

# ===== SET OPERATIONS =====
SADD tags "python" "ml" "data"   # Add to set
SMEMBERS tags                     # Get all members
SISMEMBER tags "python"           # Check membership
SCARD tags                        # Set size
SINTER set1 set2                  # Intersection
SUNION set1 set2                  # Union

# ===== SORTED SET OPERATIONS =====
ZADD leaderboard 100 "Alice" 95 "Bob" 110 "Charlie"
ZRANGE leaderboard 0 -1 WITHSCORES  # Get all with scores
ZREVRANGE leaderboard 0 2           # Top 3 (descending)
ZINCRBY leaderboard 5 "Alice"       # Increment score
ZRANK leaderboard "Alice"           # Get rank (0-indexed)
```

---

### Redis with Python

```python
import redis
import json
from datetime import timedelta

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# ===== STRING OPERATIONS =====
# Cache API response
r.setex('api:response:123', 3600, json.dumps({'data': 'value'}))

# Get cached data
cached = r.get('api:response:123')
if cached:
    data = json.loads(cached)

# Atomic counter
r.incr('page_views')

# ===== HASH OPERATIONS =====
# Store user session
r.hset('session:abc123', mapping={
    'user_id': '1001',
    'username': 'alice',
    'login_time': '2024-01-15T10:00:00'
})

# Get session data
session_data = r.hgetall('session:abc123')

# ===== LIST OPERATIONS (Queue) =====
# Producer: Add tasks to queue
r.lpush('task_queue', json.dumps({'task_id': 1, 'action': 'process'}))

# Consumer: Process tasks
task = r.rpop('task_queue')
if task:
    task_data = json.loads(task)
    # Process task...

# ===== SET OPERATIONS =====
# Track unique visitors
r.sadd('visitors:2024-01-15', 'user_1', 'user_2', 'user_3')
unique_visitors = r.scard('visitors:2024-01-15')

# ===== SORTED SET (Leaderboard) =====
# Update scores
r.zadd('game_leaderboard', {'player1': 1500, 'player2': 1200})

# Get top 10 players
top_players = r.zrevrange('game_leaderboard', 0, 9, withscores=True)
for player, score in top_players:
    print(f"{player}: {score}")

# Get player rank
rank = r.zrevrank('game_leaderboard', 'player1')  # 0-indexed

# ===== PIPELINE (Batch operations) =====
pipe = r.pipeline()
pipe.incr('counter1')
pipe.incr('counter2')
pipe.set('key1', 'value1')
results = pipe.execute()  # Execute all at once

# ===== PUB/SUB =====
# Publisher
r.publish('notifications', json.dumps({'message': 'New order!'}))

# Subscriber (in separate process/thread)
pubsub = r.pubsub()
pubsub.subscribe('notifications')
for message in pubsub.listen():
    if message['type'] == 'message':
        data = json.loads(message['data'])
        print(f"Received: {data}")
```

---

### Redis for ML Workflows

#### Use Case 1: Feature Caching

```python
import redis
import pickle
import numpy as np

r = redis.Redis(host='localhost', port=6379)

def get_user_features(user_id):
    """Get precomputed features from cache"""
    cache_key = f'features:user:{user_id}'
    
    # Try cache first
    cached = r.get(cache_key)
    if cached:
        return pickle.loads(cached)
    
    # Compute if not cached
    features = compute_features(user_id)  # Expensive operation
    
    # Cache for 1 hour
    r.setex(cache_key, 3600, pickle.dumps(features))
    return features

def compute_features(user_id):
    # Expensive feature computation...
    return np.array([0.5, 0.8, 0.3])

# Usage
features = get_user_features('U12345')
```

---

#### Use Case 2: Model Prediction Cache

```python
import redis
import json
import hashlib

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def predict_with_cache(model, input_data):
    """Cache model predictions"""
    # Create hash of input for cache key
    input_hash = hashlib.md5(
        json.dumps(input_data, sort_keys=True).encode()
    ).hexdigest()
    
    cache_key = f'prediction:{input_hash}'
    
    # Check cache
    cached_pred = r.get(cache_key)
    if cached_pred:
        return float(cached_pred)
    
    # Compute prediction
    prediction = model.predict([input_data])[0]
    
    # Cache for 10 minutes
    r.setex(cache_key, 600, str(prediction))
    return prediction
```

---

#### Use Case 3: Real-Time Leaderboard

```python
import redis

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def update_score(user_id, score_increment):
    """Update user score in leaderboard"""
    r.zincrby('game_leaderboard', score_increment, user_id)

def get_top_players(n=10):
    """Get top N players"""
    return r.zrevrange('game_leaderboard', 0, n-1, withscores=True)

def get_user_rank(user_id):
    """Get user's rank (0-indexed)"""
    rank = r.zrevrank('game_leaderboard', user_id)
    return rank + 1 if rank is not None else None

# Usage
update_score('user_123', 50)
top_10 = get_top_players(10)
rank = get_user_rank('user_123')
```

---

#### Use Case 4: Rate Limiting

```python
import redis
import time

r = redis.Redis(host='localhost', port=6379)

def rate_limit(user_id, max_requests=100, window_seconds=3600):
    """Sliding window rate limiter"""
    key = f'rate_limit:{user_id}'
    current_time = time.time()
    
    # Remove old requests outside window
    r.zremrangebyscore(key, 0, current_time - window_seconds)
    
    # Count requests in current window
    request_count = r.zcard(key)
    
    if request_count >= max_requests:
        return False  # Rate limit exceeded
    
    # Add current request
    r.zadd(key, {str(current_time): current_time})
    r.expire(key, window_seconds)
    
    return True  # Request allowed

# Usage
if rate_limit('user_123', max_requests=100, window_seconds=3600):
    # Process request
    pass
else:
    # Return 429 Too Many Requests
    pass
```

### Redis for ML

**Use cases:**
- Cache feature values for real-time inference
- Store model predictions (avoid re-computing)
- Real-time counters and metrics
- Feature flag management
- A/B test assignment

```python
import redis
import joblib

r = redis.Redis(host='localhost', port=6379)

# Load model once
model = joblib.load('model.pkl')

def predict_with_cache(user_id):
    """Real-time prediction with feature caching"""
    # Get cached features
    features_key = f'features:{user_id}'
    features = r.hgetall(features_key)
    
    if not features:
        # Compute and cache features
        features = compute_features(user_id)
        r.hmset(features_key, features)
        r.expire(features_key, 3600)  # 1 hour TTL
    
    # Prepare feature vector
    X = [float(features[f]) for f in feature_names]
    
    # Check prediction cache
    pred_key = f'pred:{user_id}'
    cached_pred = r.get(pred_key)
    if cached_pred:
        return float(cached_pred)
    
    # Predict
    prediction = model.predict([X])[0]
    
    # Cache prediction for 5 minutes
    r.setex(pred_key, 300, str(prediction))
    return prediction
```

---

### Quick Comparison: MongoDB vs Redis

| Aspect | MongoDB | Redis |
|--------|---------|-------|
| **Data Model** | Document (JSON/BSON) | Key-Value + Data Structures |
| **Storage** | Disk (with caching) | In-memory (with persistence options) |
| **Query Language** | Rich query language, aggregations | Simple key-based lookups |
| **Use in ML** | Training data, metadata, feature store | Real-time features, caching, counters |
| **Latency** | ~10-100ms | Sub-millisecond |
| **Persistence** | Durable by default | Optional (RDB snapshots, AOF logs) |
| **Scalability** | Horizontal (sharding) | Horizontal (Redis Cluster) |
| **When to Use** | Complex queries, flexible schema | Speed critical, simple data structures |

---

### Best Practices

**MongoDB:**
- ✅ Embed related data to avoid JOINs
- ✅ Create indexes on frequently queried fields
- ✅ Use aggregation pipeline for complex analytics
- ✅ Shard large collections for horizontal scaling
- ❌ Avoid deeply nested documents (>100 levels)
- ❌ Don't use MongoDB for transactions requiring strong ACID guarantees

**Redis:**
- ✅ Set appropriate TTLs to avoid memory bloat
- ✅ Use pipelines for batch operations
- ✅ Monitor memory usage (`INFO memory`)
- ✅ Use connection pooling in production
- ❌ Don't store large objects (>1MB) in Redis
- ❌ Avoid `KEYS *` in production (use `SCAN` instead)
