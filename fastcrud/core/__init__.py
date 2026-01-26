"""
FastCRUD Core Module - Centralized utilities for SQLAlchemy model operations.

This module provides the foundational utilities used throughout FastCRUD for:
- Model introspection with caching
- Data processing and transformation
- Join processing and relationship handling
- Field management and schema operations
- Configuration classes for all operations

The core module is designed with performance in mind, using strategic caching
to avoid repeated expensive operations while maintaining clean, functional APIs.
"""

from .protocols import (
    CRUDInstance,
    ModelIntrospector,
    DataProcessor,
    FilterProcessor as FilterProcessorProtocol,
    QueryBuilder,
    ResponseFormatter,
    DatabaseAdapter,
    ValidationProcessor,
)
from .introspection import ModelInspector, get_model_inspector
from .join_processing import JoinProcessor, handle_null_primary_key_multi_join
from .filtering import FilterProcessor, FilterCallable, get_sqlalchemy_filter
from .query import (
    SQLQueryBuilder,
    SortProcessor,
    JoinBuilder,
    build_joined_query,
    execute_joined_query,
    split_join_configs,
    fetch_and_merge_one_to_many,
    fetch_one_to_many_with_limit,
)

from .introspection import (
    get_primary_key_names,
    get_primary_key_columns,
    get_first_primary_key,
    get_unique_columns,
    get_python_type,
    get_column_types,
    get_sqlalchemy_column_types,
    create_composite_key,
    validate_model_has_table,
    get_model_column,
)

# Data processing module (organized by dependency level)
from .data import (
    # Data transformation functions (Level 2: pure functions)
    handle_one_to_one,
    handle_one_to_many,
    sort_nested_list,
    build_column_label,
    format_single_response,
    format_multi_response,
    create_paginated_response_data,
    convert_to_pydantic_models,
    # Data nesting functions (Level 3: uses introspection)
    nest_join_data,
    get_nested_key_for_join,
    process_joined_field,
    process_data_fields,
    cleanup_null_joins,
    # Response formatting functions (Level 4: uses join_processing)
    process_joined_data,
    format_joined_response,
)

# Pagination
from .pagination import (
    compute_offset,
    paginated_response,
    PaginatedListResponse,
    ListResponse,
    PaginatedRequestQuery,
    CursorPaginatedRequestQuery,
    create_list_response,
    create_paginated_response,
)

# Field and schema management
from .field_management import (
    create_modified_schema,
    extract_matching_columns_from_schema,
    auto_detect_join_condition,
    discover_model_relationships,
    build_relationship_joins_config,
    resolve_relationship_config,
)

# FastAPI-specific utilities
from ..fastapi_dependencies import (
    create_auto_field_injector,
    create_dynamic_filters,
    inject_dependencies,
    apply_model_pk,
)

# Configuration
from .config import (
    JoinConfig,
    CountConfig,
    CreateConfig,
    UpdateConfig,
    DeleteConfig,
    FilterConfig,
    CRUDMethods,
    validate_joined_filter_path,
)

__all__ = [
    # Protocol interfaces
    "CRUDInstance",
    "ModelIntrospector",
    "DataProcessor",
    "FilterProcessorProtocol",
    "QueryBuilder",
    "ResponseFormatter",
    "DatabaseAdapter",
    "ValidationProcessor",
    # Core classes
    "ModelInspector",
    "get_model_inspector",
    "JoinProcessor",
    "handle_null_primary_key_multi_join",
    # Filtering engine
    "FilterProcessor",
    "FilterCallable",
    "get_sqlalchemy_filter",
    # Query building engine
    "SQLQueryBuilder",
    "SortProcessor",
    "JoinBuilder",
    "build_joined_query",
    "execute_joined_query",
    "split_join_configs",
    "fetch_and_merge_one_to_many",
    "fetch_one_to_many_with_limit",
    # Introspection functions
    "get_primary_key_names",
    "get_primary_key_columns",
    "get_first_primary_key",
    "get_unique_columns",
    "get_python_type",
    "get_column_types",
    "get_sqlalchemy_column_types",
    "create_composite_key",
    "validate_model_has_table",
    "get_model_column",
    # Data transformation functions
    "handle_one_to_one",
    "handle_one_to_many",
    "sort_nested_list",
    "build_column_label",
    "format_single_response",
    "format_multi_response",
    "create_paginated_response_data",
    "convert_to_pydantic_models",
    # Data nesting functions
    "nest_join_data",
    "get_nested_key_for_join",
    "process_joined_field",
    "process_data_fields",
    "cleanup_null_joins",
    # Response formatting functions
    "process_joined_data",
    "format_joined_response",
    # Pagination utilities
    "compute_offset",
    "paginated_response",
    "PaginatedListResponse",
    "ListResponse",
    "PaginatedRequestQuery",
    "CursorPaginatedRequestQuery",
    "create_list_response",
    "create_paginated_response",
    # Field management functions
    "create_modified_schema",
    "create_auto_field_injector",
    "create_dynamic_filters",
    "extract_matching_columns_from_schema",
    "inject_dependencies",
    "apply_model_pk",
    "auto_detect_join_condition",
    "discover_model_relationships",
    "build_relationship_joins_config",
    "resolve_relationship_config",
    # Configuration classes
    "JoinConfig",
    "CountConfig",
    "CreateConfig",
    "UpdateConfig",
    "DeleteConfig",
    "FilterConfig",
    "CRUDMethods",
    "validate_joined_filter_path",
]
