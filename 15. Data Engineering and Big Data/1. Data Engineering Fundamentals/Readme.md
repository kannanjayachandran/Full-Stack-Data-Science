<h1 align="center"> Data Engineering </h1>

Data Engineering involves designing, building, and maintaining the infrastructure and systems that allow large amount of data to be collected, stored, processed, and analyzed efficiently. It involves creating pipelines that convert raw data into usable formats for analysis. The whole purpose of data engineering is to improve the business by reducing all sorts of data related overheads. 

## Data Flow

The basic flow of data in any business is as follows:

![Basic data flow](./img/basic_data_flow.png)

![Basic data flow chart](./img/data_flow.png)

> **Roles** varies according to the company and the size of the company. In a small company, a data engineer might be responsible for all the tasks mentioned above. In a large company, the tasks might be divided among different roles.

The most important takeaway from the above diagram is that data engineering sits somewhere in between the database and data science/data analysis part. 

## Data Engineering Lifecycle

The data engineering lifecycle involves the following steps:

![Data engineering life cycle](./img/life_cycle.png)

- **Data Generation** : We need data to work with. This data can be generated from various sources like applications, sensors, IOT devices, Social Media, logs, etc.

- **Data Storage** : Once we have the data, we need to store it somewhere. Depending on the type of data, size of data and other requirements such as speed of access, read/write operations, we can choose the type of storage. It can be a SQL database (relational), NoSQL database, Data Warehouses (snowflake, redshift, BigQuery, etc.), Data Lake (also called object storage (S3, Azure Blob storage, Google Cloud storage, etc.)), etc.

We have two type of data storage processing;

| **Online Transaction Processing (OLTP)** | **Online Analytical Processing (OLAP)** |
|-------------------------------------|-------------------------------------|
| Used for transactional purposes. | Used for analytical purposes.     |
| Most probably Row based, hence used for fast read/write and insert operations on row level. | Mostly column based, hence used for fast read operations on column level for large data sets. |
| Used for real-time data processing. | Used for historical data processing. |
| Examples: MySQL, PostgreSQL, etc. | Examples: Snowflake, Redshift, BigQuery, etc. |

- **Data Processing** : Once we have the data stored, we need to process it. This involves cleaning the data, transforming the data, aggregating the data, etc. or called ETL (Extract, Transform, Load) operations.

- **Data Analysis** : Once we have the processed data, we can analyze it to generate insights. This is where data science comes into play.

## Other Key Concepts

![Other aspects of data engineering](./img/other_aspects.png)

### Security

Security is a very important aspect of data engineering. We need to ensure that the data is secure and is not accessible to unauthorized users. The data should be encrypted and stored in a secure location. Additionally we may need consider data backups and disaster recovery plans.

### Data Management

It contains the following aspects:

- **Data Governance** : Data governance is the process of managing the availability, usability, integrity, and security of the data in enterprise systems, based on internal data standards and policies that also control data usage. 

- **Data Modeling** : Data modeling is the process of creating a visual representation of the data and the relationships between different data points. It helps in understanding the data and the relationships between different data points.

- **Data Integrity** : Data integrity is the maintenance of, and the assurance of the accuracy and consistency of data over its entire life-cycle. It ensures that the data is accurate and consistent throughout.

### Data Ops

DataOps is a set of practices and tools that help organizations improve the speed and quality of their data analytics. It is similar to DevOps. It involves automating the data engineering processes, monitoring the data pipelines, and ensuring that the data is accurate and consistent.**(Data governance, Observability, Monitoring and Incident reporting)**

### Data Architecture

Data architecture is the design of the data infrastructure that supports the data engineering processes. It consists of designing the data storage, data processing, data analysis components of the data infrastructure and designing the data pipelines that move the data from one component to another.

### Data Orchestration

Data orchestration is the process of automating the data engineering processes. It involves scheduling the data pipelines, monitoring the data pipelines, and ensuring that the data pipelines are running in order and timely.

### Software Engineering

Data engineering involves a lot of software engineering practices. It involves writing good quality code to process the data, writing code to automate the data engineering processes, writing code to monitor the data pipelines, etc.

## Data Architecture

Data architecture is the design of systems that support the evolving data need of an organization, achieved by flexible and reversible decisions reached through a careful evaluation of trade-offs. It encompasses data models, policies, standards, and the overall structure for data flow within an organization.

The following diagram shows an high-level overview of data architecture.

![Generic data architecture](./img/data_arch.png)

Data architecture has two main components:

- Business side or Operational architecture

- Technical architecture

### Business Side

Operational architecture ensures that the data practices aligns closely with your business needs. It should govern every piece of data you collect, store, and process. Some of the key considerations for business side are:

- **Start with the end in mind**: Understand the business goals  or objectives and design the data architecture to support those goals.

- **Iterate and improve**: Data architecture is not a one-time thing. It should evolve as the business evolves.

- **Focus on impact**: Focus on the data that has the most impact on the business.

### Technical Architecture

Technical architecture talks about the actual tools and technologies we use to build these data infrastructure. Selecting the right tools and technologies is very important. Some of the key considerations for technical architecture are:

1. Keep the architecture simple while meeting the business needs; no need to over-engineer the solution or use the latest and greatest tools.

2. Choose the right tool for the right job; there is no one-size-fits-all solution.

3. Build for scale and flexibility; businesses pivot and change, so should the data architecture.

4. Embrace automation.

5. Prioritize data security and governance/compliance.

Some of the key components of the technical architecture are:

![Technical architecture](./img/technical_arch.png)

A typical data architecture using services from AWS is shown below:

![AWS based data architecture](./img/aws_arch.png)

## Data Warehouse

Data Warehouse is a centralized repository of integrated data from various sources, optimized for analysis and reporting. It stores historical data in a structured format, supporting business intelligence and decision-making processes. We follow the `ETL` process to load data into the data warehouse.

![ETL process](./img/ETL.png)

Some organizations may prefer `ELT` process where data is loaded into the data warehouse first and then transformed. This may not be suitable always, as real world data is messy and almost always requires some level of cleaning and transformation before it can be loaded into the data warehouse.

![ELT Process](./img/ELT.png)

### Dimensional Modeling

Dimensional modeling is a design technique for databases intended to support end-user queries in a data warehouse. It optimizes the database for fast queries and easy access to data. It is based on two types of tables:

- **Fact Tables**: These tables contain the quantitative data that can be aggregated. They are typically large and contain foreign keys to dimension tables.

- **Dimension Tables**: These tables contain the descriptive data that provides context to the fact data. They are typically smaller and contain textual data.

We can use `Star Schema` or `Snowflake Schema` for dimensional modeling. Where in star schema, the fact table is at the center and the dimension tables are connected to the fact table. It is called a star schema because it looks like a star. In snowflake schema, the dimension tables are normalized into multiple related tables, which reduces redundancy and improves data integrity.

![Schemas](./img/schema.png)

### Slowly Changing Dimensions

Slowly changing dimensions are dimensions that change slowly over time. There are three types of slowly changing dimensions:

- **Type 1**: In this type, the old data is simply overwritten with the new data. This is the simplest method but it does not keep track of the history.

> EG: If the city of a customer changes, we simply update the city in the customer table. 

- **Type 2**: In this type, a new row is added to the dimension table whenever the data changes. This keeps track of the history but can lead to a large number of rows in the dimension table.

> EG: If the city of a customer changes, we add a new row to the customer table with the new city and a new primary key.

- **Type 3**: In this type, a new column is added to the dimension table to store the new data. This keeps track of the history but can lead to a large number of columns in the dimension table.

> EG: If the city of a customer changes, we add a new column to the customer table to store the new city.

