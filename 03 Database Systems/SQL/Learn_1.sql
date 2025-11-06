-- Creating a table
CREATE TABLE customer(
first_name VARCHAR(30) NOT NULL,
last_name VARCHAR(30) NOT NULL,
email VARCHAR(60) NOT NULL,
company VARCHAR(60) NOT NULL,
street VARCHAR(60) NOT NULL,
city VARCHAR(60) NOT NULL,
state CHAR(2) NOT NULL,
pin_code SMALLINT NOT NULL,
phone_number VARCHAR(20) NOT NULL,
birth_date DATE NOT NULL,
sex CHAR(1) NOT NULL,
date_entered TIMESTAMP NOT NULL,
id SERIAL PRIMARY KEY NOT NULL
);
-- SERIAL is auto incremented and it is the preferred datatype for id's.

-- Inserting data into the table
INSERT INTO customer(first_name, last_name, email, company, street, 
					city, state, pin_code, phone_number, birth_date,
					sex, date_entered)
VALUES('Adam', 'John', 'adamjohn123@gmail.com', 'Dawn', 'second Street',
	  'New city', 'PU', 645, '123-456-987', '1980-12-21', 'M', 
	   current_timestamp);

-- Creating Custom data-types
CREATE TYPE sex_type AS ENUM ('M', 'F')

-- Altering the table
ALTER TABLE customer
ALTER COLUMN sex TYPE sex_type USING sex::sex_type;

-- Table for sales_person
CREATE TABLE sales_person(
first_name VARCHAR(30) NOT NULL,
last_name VARCHAR(30) NOT NULL,
email VARCHAR(60) NOT NULL,
company VARCHAR(60) NOT NULL,
street VARCHAR(60) NOT NULL,
city VARCHAR(60) NOT NULL,
state CHAR(2) NOT NULL,
pin_code SMALLINT NOT NULL,
phone_number VARCHAR(20) NOT NULL,
birth_date DATE NOT NULL,
sex sex_type NOT NULL,
date_hired TIMESTAMP NOT NULL,
id SERIAL PRIMARY KEY NOT NULL
);

-- Create product_type table
CREATE TABLE product_type(
name VARCHAR(40) NOT NULL,
id SERIAL PRIMARY KEY
);

-- Creating a product table
CREATE TABLE product (
type_id INTEGER REFERENCES product_type(id),
name VARCHAR(40) NOT NULL,
supplier VARCHAR(40) NOT NULL,
description TEXT NOT NULL,
id SERIAL PRIMARY KEY
);
