# Advanced Usage Overview

The Advanced section of our documentation delves into the sophisticated capabilities and features of our application, tailored for users looking to leverage advanced functionalities. This part of our guide aims to unlock deeper insights and efficiencies through more complex use cases and configurations.

## Key Topics

### 1. Advanced Filtering and Searching

Explore how to implement advanced filtering and searching capabilities in your application. This guide covers the use of comparison operators (such as greater than, less than, etc.), pattern matching, logical operations like OR and NOT, and more to perform complex queries.

- [Advanced Filtering Guide](filters.md)
- [Legacy Filtering Options](crud.md#advanced-filters)

### 2. Bulk Operations and Batch Processing

Learn how to efficiently handle bulk operations and batch processing. This section provides insights into performing mass updates, deletes, and inserts, optimizing performance for large datasets.

- [Bulk Operations Guide](crud.md#allow-multiple-updates-and-deletes)

### 3. Soft Delete Mechanisms and Strategies

Understand the implementation of soft delete mechanisms within our application. This guide covers configuring and using custom columns for soft deletes, restoring deleted records, and filtering queries to exclude soft-deleted entries.

- [Soft Delete Strategies Guide](endpoint.md#custom-soft-delete)

### 4. Advanced Use of EndpointCreator and crud_router

This topic extends the use of `EndpointCreator` and `crud_router` for advanced endpoint management, including creating custom routes, selective method exposure, and integrating soft delete functionalities.

- [Advanced Endpoint Management Guide](endpoint.md#advanced-use-of-endpointcreator)

### 5. Using `get_joined` and `get_multi_joined` for multiple models

Explore the use of `get_joined` and `get_multi_joined` functions for complex queries that involve joining multiple models, including self-joins and scenarios requiring multiple joins on the same model.

- [Joining Multiple Models Guide](crud.md#using-get_joined-and-get_multi_joined-for-multiple-models)

### 6. Method Chaining with `select`

FastCRUD's `select` method introduces method chaining, allowing for the construction of detailed queries with a focus on precision. It simplifies the process of dynamically applying filters, sorting, and conditions, making it easier to manage complex query requirements.

- [Method Chaining Guide](crud.md#enhanced-query-capabilities-with-method-chaining)

### 7. In depth explanation of Joined methods

Explore different ways of joining models in FastCRUD with examples and tips.

- [Joining Models](joins.md#applying-joins-in-fastcrud-methods)

### 8. Filters in Automatic Endpoints

Learn how to add query parameters to your `get_multi` and `get_paginated` endpoints.

- [Filters in Endpoints](endpoint.md#defining-filters)

### 9. Response Key Configuration

Customize the key used for list responses in FastCRUD through the `multi_response_key` parameter. This guide covers default behavior and how to configure custom response keys for better API consistency.

- [Response Key Configuration Guide](response_keys.md)

### 10. Row-Level Policies

Decide which operations a caller may perform and which rows and columns each one reaches with `PolicyConfig`, in the manner of Postgres row-level security, directly on `FastCRUD` or per request in `crud_router`.

- [Row-Level Policies Guide](policies.md)

## Prerequisites

Advanced usage assumes a solid understanding of the basic features and functionalities of our application. Knowledge of FastAPI, SQLAlchemy, and Pydantic is highly recommended to fully grasp the concepts discussed.
