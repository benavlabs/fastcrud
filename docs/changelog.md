# FastCRUD Changelog

## Introduction

The Changelog documents all notable changes made to FastCRUD. This includes new features, bug fixes, and improvements. It's organized by version and date, providing a clear history of the library's development.
___

## [0.20.0] - Dec 13, 2025

#### Added
- **Custom Filter Operators** by [@igorbenav](https://github.com/igorbenav)
  - Define and use custom filter operators at the instance level via `custom_filters` parameter
  - Custom filters extend or override built-in operators like `eq`, `gt`, `like`, etc.
  - Instance-level isolation allows different `FastCRUD` instances to have different filter behaviors
  - New `FilterCallable` type exported for type hints
  - Works with `FastCRUD`, `EndpointCreator`, and `crud_router`

#### Fixed
- **Swagger Operation Names** by [@igorbenav](https://github.com/igorbenav)
  - Auto-generated CRUD endpoints now display meaningful operation IDs in OpenAPI documentation
  - Endpoints show `modelname_create`, `modelname_read`, `modelname_read_multi`, etc. instead of generic "Endpoint"
  - Added `name` parameter to all `add_api_route()` calls in `EndpointCreator`
- **TypedDict Response Types** by [@igorbenav](https://github.com/igorbenav)
  - `GetMultiResponseModel` and `GetMultiResponseDict` now use proper `TypedDict` classes
  - Eliminates type checker false positives when accessing `data` or `total_count` keys
  - Added `UpsertMultiResponseDict` and `UpsertMultiResponseModel` TypedDict classes

#### Improved
- **Python Version Upgrade** by [@igorbenav](https://github.com/igorbenav)
  - Minimum Python version upgraded from 3.9 to 3.11
  - Enables use of modern Python features and improved performance

#### Removed
- **`fastcrud.paginated` Module** by [@igorbenav](https://github.com/igorbenav)
  - The deprecated `fastcrud.paginated` module has been completely removed
  - Import pagination utilities directly from `fastcrud` instead:
    ```python
    from fastcrud import PaginatedListResponse, PaginatedRequestQuery
    ```

#### Breaking Changes
⚠️ **`create()` Method Return Type Changed**

The `create()` method no longer returns the SQLAlchemy model when `schema_to_select` is not provided. It now returns `None` for consistency with other CRUD methods.

```python
# Before (v0.19.x) - returned SQLAlchemy model
user = await crud.create(db, user_data)
print(user.id)  # Worked

# After (v0.20.0) - returns None
result = await crud.create(db, user_data)
print(result)  # None

# To get data back, provide schema_to_select
user = await crud.create(db, user_data, schema_to_select=UserSchema, return_as_model=True)
```

⚠️ **`fastcrud.paginated` Module Removed**

Update imports from deprecated module:
```python
# Old (no longer works)
from fastcrud.paginated import PaginatedListResponse

# New (required)
from fastcrud import PaginatedListResponse
```

⚠️ **Python 3.11+ Required**

Python 3.9 and 3.10 are no longer supported.

#### What's Changed
* Remove deprecated by [@igorbenav](https://github.com/igorbenav) in https://github.com/benavlabs/fastcrud/pull/292
* Some features and fixes by [@igorbenav](https://github.com/igorbenav) in https://github.com/benavlabs/fastcrud/pull/298

**Full Changelog**: https://github.com/benavlabs/fastcrud/compare/v0.19.2...v0.20.0

___

## [0.19.2] - Nov 15, 2025

#### Added
- **get_joined Method Overloads** by [@igorbenav](https://github.com/igorbenav)
  - Added missing `@overload` signatures for `get_joined()` method to support proper type inference
  - Added `return_as_model` parameter for converting joined results to Pydantic models
  - Enhanced type safety: returns `SelectSchemaType` when `return_as_model=True`, `dict` when `False`
  - Consistent API with other CRUD methods like `get()`, `update()`, etc.

#### Improved  
- **create Method Performance and Consistency** by [@igorbenav](https://github.com/igorbenav)
  - Removed unnecessary database round-trip by eliminating redundant `get()` call after creation
  - Added deprecation warnings for upcoming API consistency changes in next major version
  - Fixed type hints to match actual implementation behavior (removed incorrect `None` return type)

#### Deprecated
- **create Method Behavior Changes** (Warnings Added)
  - `create()` without `schema_to_select` will return `None` instead of SQLAlchemy model in next major version
  - `create()` with `schema_to_select` will properly respect `return_as_model` parameter in next major version
  - These changes align `create()` behavior with `update()` and other CRUD methods for consistency

#### Fixed
- **Type Safety Issues** by [@igorbenav](https://github.com/igorbenav)
  - Fixed `get_joined()` method type annotations to properly reflect actual return types
  - Corrected `create()` method type hints to remove impossible `None` return type
  - Enhanced test coverage with 8 new comprehensive tests for `get_joined` return type variations

#### Breaking Changes  
⚠️ **None** - This release maintains full backward compatibility with 0.19.1. Deprecation warnings provide clear migration path for next major version.

___

## [0.19.1] - Nov 10, 2025

#### Improved
- **Enhanced Type Safety for Schema-based Column Selection** by [@igorbenav](https://github.com/igorbenav)
  - Added missing `@overload` signatures for `schema_to_select` with `return_as_model=False`
  - Eliminates union types and manual casting when using schema-based column selection with dictionary returns
  - Enhanced 8 methods: `create()`, `get()`, `update()`, `upsert()`, `upsert_multi()`, `get_multi()`, `get_multi_by_cursor()`, `get_multi_joined()`
  - Provides precise type inference for the common pattern of column selection with dict access

#### Fixed
- **Documentation Link** by [@igorbenav](https://github.com/igorbenav)
  - Fixed broken link in project documentation

#### Breaking Changes
⚠️ **None** - This release maintains full backward compatibility with 0.19.0

#### What's Changed
* fix link by [@igorbenav](https://github.com/igorbenav) in https://github.com/benavlabs/fastcrud/pull/284
* Overload for select_schema but not return_as_model by [@igorbenav](https://github.com/igorbenav) in https://github.com/benavlabs/fastcrud/pull/285

**Full Changelog**: https://github.com/benavlabs/fastcrud/compare/v0.19.0...v0.19.1

___

## [0.19.0] - Nov 7, 2025

#### Added
- **Complete Architectural Overhaul** by [@igorbenav](https://github.com/igorbenav)
  - Six-layer protocol-based architecture replacing helper-based design
  - 8 new Protocol interfaces eliminating circular dependencies and enabling clean testing
  - 15+ specialized core modules with strategic caching and performance optimizations
  - Enhanced type safety with method overloads and `return_as_model` parameter across 7 methods
- **New Configuration Classes** by [@igorbenav](https://github.com/igorbenav)
  - `CreateConfig`, `UpdateConfig`, `DeleteConfig` for server-side field injection and security
  - Auto field injection with dependency support and schema exclusion capabilities
  - All configuration classes available from main `fastcrud` import
- **Enhanced Type System** by [@igorbenav](https://github.com/igorbenav)
  - New precise types: `FilterValueType`, `FilterValue`, `FilterValueSequence`
  - Response types: `GetMultiResponseDict`, `GetMultiResponseModel`, `UpsertMultiResponseDict`, `UpsertMultiResponseModel`
  - Full type safety with proper type guards and isinstance checks

#### Fixed
- **Cartesian Product in Complex Joins** by [@igorbenav](https://github.com/igorbenav)
  - Fixed duplicate results in `get_joined` with multiple one-to-many joins and `nest_joins=True`
  - Enhanced join processing with proper deduplication for complex relationship scenarios

#### Improved
- **Performance Optimizations** by [@igorbenav](https://github.com/igorbenav)
  - Instance-level caching for SQLAlchemy introspection operations
  - Multi-level caching implementation for query building and schema modifications
  - Pure function architecture patterns for data transformation
- **Enhanced Import Structure** by [@igorbenav](https://github.com/igorbenav)
  - All utilities now available from main `fastcrud` import
  - Deprecated `fastcrud.paginated` module with proper warning system
  - Centralized configuration and utility exports

#### Breaking Changes
⚠️ **None** - This release maintains full backward compatibility with 0.18.x

#### Deprecated
- **fastcrud.paginated Module** - Import pagination utilities directly from `fastcrud` instead

#### What's Changed
* Major refactor by [@igorbenav](https://github.com/igorbenav) in https://github.com/benavlabs/fastcrud/pull/275
* Get joined fix by [@igorbenav](https://github.com/igorbenav) in https://github.com/benavlabs/fastcrud/pull/276  
* FastCRUD refactor by [@igorbenav](https://github.com/igorbenav) in https://github.com/benavlabs/fastcrud/pull/277
* Architecture refactor by [@igorbenav](https://github.com/igorbenav) in https://github.com/benavlabs/fastcrud/pull/278
* Better typing by [@igorbenav](https://github.com/igorbenav) in https://github.com/benavlabs/fastcrud/pull/279
* Fix codecoverage by [@igorbenav](https://github.com/igorbenav) in https://github.com/benavlabs/fastcrud/pull/280
* Final preparations for 0.19.0 by [@igorbenav](https://github.com/igorbenav) in https://github.com/benavlabs/fastcrud/pull/281

**Full Changelog**: https://github.com/benavlabs/fastcrud/compare/v0.18.1...v0.19.0

___

## [0.18.1] - Nov 4, 2025

#### Fixed
- **Composite Primary Key Deduplication Bug in Joins** by [@igorbenav](https://github.com/igorbenav)
  - Fixed critical bug where models with composite primary keys were incorrectly deduplicated in one-to-many relationships
  - Enhanced primary key handling to support full composite key comparison instead of just the first key
  - Comprehensive refactoring of join helper functions for better maintainability and performance
  - Added extensive test coverage for composite primary key scenarios

#### Improved
- **Join Processing Architecture** by [@igorbenav](https://github.com/igorbenav)
  - Major refactoring of `fastcrud/crud/helper.py` with improved code organization
  - Enhanced column extraction and labeling logic for better join performance
  - Improved error handling and validation for model attributes
  - Better separation of concerns in join configuration processing

#### Breaking Changes
⚠️ **None** - This release maintains full backward compatibility with 0.18.0

#### What's Changed
* fix composite key bug in joins, refactor helper by [@igorbenav](https://github.com/igorbenav)
* change version by [@igorbenav](https://github.com/igorbenav)

**Full Changelog**: https://github.com/benavlabs/fastcrud/compare/v0.18.0...v0.18.1

___

## [0.18.0] - Nov 4, 2025

#### Added
- **CountConfig for Related Object Counting** by [@doubledare704](https://github.com/doubledare704)
  - New configuration class for efficiently counting related objects in joined queries
  - Supports many-to-many and one-to-many relationship counting via scalar subqueries
  - Comprehensive support for composite primary keys and complex joins
  - Seamless integration with `get_multi_joined` method
- **PaginatedRequestQuery Schema** by [@doubledare704](https://github.com/doubledare704)
  - Reusable Pydantic schema for standardizing pagination query parameters
  - Supports both page-based and offset-based pagination modes
  - Enhanced OpenAPI documentation with proper field descriptions
  - Backward-compatible implementation with existing endpoints
- **Server-Side Field Injection (Pre-Processor Functions)** by [@LucasQR](https://github.com/LucasQR)
  - Automatic field injection for create, update, and delete operations
  - CreateConfig, UpdateConfig, and DeleteConfig for comprehensive endpoint configuration
  - Support for dependency injection with FastAPI's Depends system
  - Security-focused field exclusion from API schemas and documentation

#### Fixed
- **return_as_model Join Prefix Compatibility** by [@igorbenav](https://github.com/igorbenav)
  - Resolved silent failure when join_prefix doesn't match Pydantic schema field names
  - Added comprehensive validation with clear error messages for mismatched configurations
  - Enhanced support for nested joins with proper field mapping
  - Prevents data loss in joined relationships when using return_as_model=True

#### Improved
- **Documentation and Community** by [@emiliano-gandini-outeda](https://github.com/emiliano-gandini-outeda), [@LucasQR](https://github.com/LucasQR), [@igorbenav](https://github.com/igorbenav)
  - Added DeepWiki badge and documentation links to README
  - Fixed Discord invite to use permanent link instead of temporary
  - Enhanced join documentation with compatibility warnings and examples
  - Community features and improved project accessibility

#### Breaking Changes
⚠️ **None** - This release maintains full backward compatibility with 0.17.x

**Full Changelog**: https://github.com/benavlabs/fastcrud/compare/v0.17.1...v0.18.0

___

## [0.17.1] - Oct 8, 2025

#### Added
- **Multiple Values Support for OR/NOT Filters** by [@igorbenav](https://github.com/igorbenav)
  - Enhanced `__or` and `__not` operators to accept lists of values for the same operator
  - Enables multiple LIKE patterns, equality checks, and other operators in single filter conditions
  - Syntax: `name__or={"like": ["Alice%", "Frank%"]}`
  - Fully backward compatible with existing filter syntax
  - Comprehensive test coverage for both SQLAlchemy and SQLModel

#### Fixed
- **Dictionary Key Duplication in Tests** by [@igorbenav](https://github.com/igorbenav)
  - Fixed invalid Python syntax in test files where dictionary keys were duplicated
  - Updated existing tests to use new list syntax for multiple values
  - Resolved ruff linting errors in test suite

#### Improved
- **Filter Documentation Enhancement** by [@igorbenav](https://github.com/igorbenav)
  - Updated advanced filters documentation with new list syntax examples
  - Added comprehensive usage patterns for multiple value filtering
  - Enhanced examples showing mixed list and single value usage

#### Breaking Changes
⚠️ **None** - This release maintains full backward compatibility with 0.17.x

#### What's Changed
* feat: support multiple values for same operator in OR/NOT filters by [@igorbenav](https://github.com/igorbenav)
* fix: resolve duplicate dictionary keys in test files by [@igorbenav](https://github.com/igorbenav)
* docs: enhance filtering documentation with list syntax examples by [@igorbenav](https://github.com/igorbenav)
* test: add comprehensive coverage for multiple value filtering by [@igorbenav](https://github.com/igorbenav)

**Full Changelog**: https://github.com/benavlabs/fastcrud/compare/v0.17.0...v0.17.1

___

## [0.17.0] - Sep 25, 2025

#### Fixed
- **Pydantic Relationship Fields Issue** by [@doubledare704](https://github.com/doubledare704)
  - Fixed `get_multi` returning boolean values for Pydantic relationship fields when select schema is provided
  - Resolved cartesian product issues when including relationship fields in schemas
  - Enhanced test coverage with comprehensive SQL verification
- **Delete Methods Filtering & Typing** by [@doubledare704](https://github.com/doubledare704)
  - Added missing filter support for delete operations
  - Improved type annotations for delete methods
  - Enhanced type safety for delete operations
- **Dynamic Filter Parsing** by [@luminosoda](https://github.com/luminosoda)
  - Fixed parsing issue with filter operators containing double underscores
  - Correctly handles complex filter operations like `field__gte`, `field__lt`
  - Improved robustness of dynamic filter processing

#### Added
- **Joined Model Filtering for Automatic Endpoints** by [@doubledare704](https://github.com/doubledare704)
  - Support for dot notation in filter configurations (e.g., `company.name`)
  - Automatic detection and handling of relationship-based filters
  - Enhanced FilterConfig to validate and parse joined model filters
  - Comprehensive filtering across related models in CRUD endpoints

#### Improved
- **Test Infrastructure Enhancement** by [@doubledare704](https://github.com/doubledare704)
  - Refactored test suite to eliminate pytest warnings
  - Enhanced test coverage with SQL-level verification
  - Better integration with existing test patterns
  - More descriptive test naming conventions

#### Documentation Updates
- **Joined Model Filtering Guide** with comprehensive examples and usage patterns
- **Enhanced Endpoint Documentation** with detailed filtering examples
- **Improved Test Documentation** with better patterns and conventions

#### Breaking Changes
⚠️ **None** - This release maintains full backward compatibility with 0.16.x

#### What's Changed
* add analytics to mkdocs by [@igorbenav](https://github.com/igorbenav) in [#250](https://github.com/benavlabs/fastcrud/pull/250)
* 🔧Fix for get_multi returns boolean values for pydantic relationship fields, if a select schema is provided #199 by [@doubledare704](https://github.com/doubledare704) in [#245](https://github.com/benavlabs/fastcrud/pull/245)
* fix: add delete filters and type annotation for delete methods. #147 by [@doubledare704](https://github.com/doubledare704) in [#244](https://github.com/benavlabs/fastcrud/pull/244)
* feat: implement joined model filtering for automatic endpoints by [@doubledare704](https://github.com/doubledare704) in [#246](https://github.com/benavlabs/fastcrud/pull/246)
* Fix #248 by [@luminosoda](https://github.com/luminosoda) in [#249](https://github.com/benavlabs/fastcrud/pull/249)
* update pyproject by [@igorbenav](https://github.com/igorbenav) in [#252](https://github.com/benavlabs/fastcrud/pull/252)

#### New Contributors
* [@luminosoda](https://github.com/luminosoda) made their first contribution in [#249](https://github.com/benavlabs/fastcrud/pull/249)

**Full Changelog**: https://github.com/benavlabs/fastcrud/compare/v0.16.0...v0.17.0

___

## [0.16.0] - Aug 25, 2025

#### Added
- **Enhanced Create Method** by [@igorbenav](https://github.com/igorbenav)
  - Added `schema_to_select` parameter for selecting specific columns
  - Added `return_as_model` to return Pydantic models instead of SQLAlchemy instances
  - Flexible data return methods for better API design
- **Advanced Sorting Functionality** by [@igorbenav](https://github.com/igorbenav)
  - Multi-field sorting support with ascending/descending control
  - Flexible sorting syntax like `field1,-field2`
  - Enhanced query performance and user experience
- **Dependency-Based Filtering** by [@igorbenav](https://github.com/igorbenav)
  - Runtime filtering using FastAPI dependencies
  - Supports row-level access control
  - Seamless authentication system integration

#### Improved
- **Enhanced Documentation** with comprehensive examples and usage patterns
- **Better SQLAlchemy Support** for non-native column types
- **Type Safety Enhancements** across the codebase
- **Performance Optimizations** for query handling

#### Breaking Changes
⚠️ **None** - This release maintains full backward compatibility with 0.15.x

#### What's Changed
* Enhanced create method with flexible return options by [@igorbenav](https://github.com/igorbenav)
* Advanced sorting functionality implementation by [@igorbenav](https://github.com/igorbenav)
* Dependency-based filtering for enhanced security by [@igorbenav](https://github.com/igorbenav)
* Documentation improvements and examples by [@igorbenav](https://github.com/igorbenav)

**Full Changelog**: https://github.com/benavlabs/fastcrud/compare/v0.15.12...v0.16.0

___

## [0.15.12] - Jun 9, 2025

#### Added
- **Configurable Response Key** by [@igorbenav](https://github.com/igorbenav)
  - Added configurable response key for `get_multi` method
  - Enhanced flexibility in API response structure
- **Documentation Improvements** by [@igorbenav](https://github.com/igorbenav)
  - Moved documentation to new location
  - Added banner and CRUDAdmin mention

#### Improved
- **Dependency Updates** by [@arab0v](https://github.com/arab0v)
  - Updated SQLAlchemy-utils dependency to version 0.41.2
  - Enhanced compatibility and security

#### What's Changed
* Added banner and CRUDAdmin mention by [@igorbenav](https://github.com/igorbenav)
* Updated SQLAlchemy-utils dependency to version 0.41.2 by [@arab0v](https://github.com/arab0v)
* Added configurable response key for get_multi method by [@igorbenav](https://github.com/igorbenav)
* Moved documentation by [@igorbenav](https://github.com/igorbenav)
* Bumped project version by [@igorbenav](https://github.com/igorbenav)

#### New Contributors
* [@arab0v](https://github.com/arab0v) made their first contribution

**Full Changelog**: https://github.com/benavlabs/fastcrud/compare/v0.15.11...v0.15.12

___

## [0.15.11] - May 10, 2025

#### Added
- **Multi-Field OR Filter Functionality** by [@doubledare704](https://github.com/doubledare704)
  - Implemented multi-field OR filter functionality
  - Enhanced querying capabilities across multiple fields
  - Improved filter flexibility for complex search scenarios

#### What's Changed
* Add multi-field OR filter functionality by [@doubledare704](https://github.com/doubledare704)
* Preparations for 0.15.11 by [@igorbenav](https://github.com/igorbenav)

**Full Changelog**: https://github.com/benavlabs/fastcrud/compare/v0.15.10...v0.15.11

___

## [0.15.10] - May 9, 2025

#### Fixed
- **Metadata Publishing Bug** by [@igorbenav](https://github.com/igorbenav)
  - Fixed bug in `pyproject.toml` that caused versions 0.15.8 and 0.15.9 to be published only with metadata
  - Resolved packaging issues for proper distribution

#### Notes
Versions `0.15.8` and `0.15.9` were published only with metadata because of a bug in `pyproject.toml`. This version resolves the packaging issue.

**Full Changelog**: https://github.com/benavlabs/fastcrud/compare/v0.15.9...v0.15.10

___

## [0.15.9] - May 9, 2025

#### Notes
This version was published only with metadata due to a bug in `pyproject.toml`. See version 0.15.10 for the fix.

**Full Changelog**: https://github.com/benavlabs/fastcrud/compare/v0.15.8...v0.15.9

___

## [0.15.8] - May 9, 2025

#### Added
- **New Documentation Page** by [@igorbenav](https://github.com/igorbenav)
  - Added comprehensive documentation page
  - Enhanced project documentation structure
- **UV Package Manager** by [@VDuchauffour](https://github.com/VDuchauffour)
  - Introduced UV as package manager
  - Improved development workflow and dependency management

#### Fixed
- **get_multi_joined Total Count** by [@igorbenav](https://github.com/igorbenav)
  - Fixed `get_multi_joined` total_count issue with join_model parameter
  - Resolved counting inconsistencies in joined queries
- **get_multi_by_cursor Bug** by [@igorbenav](https://github.com/igorbenav)
  - Fixed code issues in `get_multi_by_cursor` method
  - Improved cursor-based pagination reliability

#### Improved
- **Code Optimizations** by [@igorbenav](https://github.com/igorbenav)
  - Performed code optimizations in fast_crud.py
  - Enhanced performance and maintainability
- **Sorting for Nested Fields** by [@igorbenav](https://github.com/igorbenav)
  - Implemented sorting functionality for nested fields
  - Enhanced query capabilities for complex data structures
- **Project Configuration** by [@igorbenav](https://github.com/igorbenav)
  - Updated pyproject.toml configuration
  - Updated README.md with latest information

#### What's Changed
* Added new documentation page by [@igorbenav](https://github.com/igorbenav)
* Updated pyproject.toml by [@igorbenav](https://github.com/igorbenav)
* Fixed `get_multi_joined` total_count with join_model parameter by [@igorbenav](https://github.com/igorbenav)
* Code optimizations in fast_crud.py by [@igorbenav](https://github.com/igorbenav)
* Introduced UV as package manager by [@VDuchauffour](https://github.com/VDuchauffour)
* Fixed code in `get_multi_by_cursor` by [@igorbenav](https://github.com/igorbenav)
* Implemented sorting for nested fields by [@igorbenav](https://github.com/igorbenav)
* Updated README.md by [@igorbenav](https://github.com/igorbenav)

#### New Contributors
* [@VDuchauffour](https://github.com/VDuchauffour) made their first contribution

**Full Changelog**: https://github.com/benavlabs/fastcrud/compare/v0.15.7...v0.15.8

___

## [0.15.7] - Mar 25, 2025

#### Added
- **Advanced filter configs** by [@doubledare704](https://github.com/doubledare704)
- **OR and NOT for filtering** by [@doubledare704](https://github.com/doubledare704)

#### Improved
- **Remove redundant code** by [@suhanwu](https://github.com/suhanwu) in [#211](https://github.com/igorbenav/fastcrud/pull/211)
- **Added pragma: no cover to relevant lines** by [@igorbenav](https://github.com/igorbenav) in [#212](https://github.com/igorbenav/fastcrud/pull/212)

#### What's Changed
* Implement advanced filter configs by [@doubledare704](https://github.com/doubledare704) in [#204](https://github.com/igorbenav/fastcrud/pull/204)
* Implement OR and NOT for filtering by [@doubledare704](https://github.com/doubledare704) in [#210](https://github.com/igorbenav/fastcrud/pull/210)
* Fix: Remove redundant code by [@suhanwu](https://github.com/suhanwu) in [#211](https://github.com/igorbenav/fastcrud/pull/211)
* Added # pragma: no cover to relevant lines by [@igorbenav](https://github.com/igorbenav) in [#212](https://github.com/igorbenav/fastcrud/pull/212)

#### New Contributors
* [@suhanwu](https://github.com/suhanwu) made their first contribution in [#211](https://github.com/igorbenav/fastcrud/pull/211)
* [@doubledare704](https://github.com/doubledare704) made their first contribution in [#204](https://github.com/igorbenav/fastcrud/pull/204)

**Full Changelog**: https://github.com/igorbenav/fastcrud/compare/v0.15.6...v0.15.7
___
## [0.15.6] - Feb 19, 2025

#### Added
- **Models and Schemas, Batch 5: Projects and Participants** by [@slaarti](https://github.com/slaarti) in [#195](https://github.com/igorbenav/fastcrud/pull/195)
- **Security.md** by [@igorbenav](https://github.com/igorbenav) in [#200](https://github.com/igorbenav/fastcrud/pull/200)

#### Fixed
- **Cryptography Package Vulnerability** updated to address OpenSSL vulnerability by [@igorbenav](https://github.com/igorbenav) in [#202](https://github.com/igorbenav/fastcrud/pull/202)

#### Documentation Updates
- **Showcase** by [@igorbenav](https://github.com/igorbenav) in [#193](https://github.com/igorbenav/fastcrud/pull/193)

#### What's Changed
* Showcase by [@igorbenav](https://github.com/igorbenav) in [#193](https://github.com/igorbenav/fastcrud/pull/193)
* Models and Schemas, Batch 5: Projects and Participants by [@slaarti](https://github.com/slaarti) in [#195](https://github.com/igorbenav/fastcrud/pull/195)
* Create SECURITY.md by [@igorbenav](https://github.com/igorbenav) in [#200](https://github.com/igorbenav/fastcrud/pull/200)
* Bump cryptography to fix vulnerability by [@igorbenav](https://github.com/igorbenav) in [#202](https://github.com/igorbenav/fastcrud/pull/202)

**Full Changelog**: https://github.com/igorbenav/fastcrud/compare/v0.15.5...v0.15.6
___
## [0.15.5] - Dec 25, 2024

#### Fixed
- **Get multi with return_as_model** is now properly typed
- **Filter with a UUID** that is not a primary key now working
- **Update with not found record** now raises error as previously defined by warning
- **Response model** working properly in swagger

#### What's Changed
* Some fixes by [@igorbenav](https://github.com/igorbenav) in [#190](https://github.com/igorbenav/fastcrud/pull/190)
* Response model in swagger by [@igorbenav](https://github.com/igorbenav) in [#191](https://github.com/igorbenav/fastcrud/pull/191)
* Change version in pyproject to 0.15.5 by [@igorbenav](https://github.com/igorbenav) in [#192](https://github.com/igorbenav/fastcrud/pull/192)

**Full Changelog**: https://github.com/igorbenav/fastcrud/compare/v0.15.4...v0.15.5
___
## [0.15.4] - Dec 23, 2024

#### Added
- **Implement select_schema on EndpointCreator and crud_router** by [@ljmc-github](https://github.com/ljmc-github) in [#169](https://github.com/igorbenav/fastcrud/pull/169)

#### Fixed
- **Custom name bug fix** by [@igorbenav](https://github.com/igorbenav) in [#187](https://github.com/igorbenav/fastcrud/pull/187)
- **UUID support fix** by [@igorbenav](https://github.com/igorbenav) in [#188](https://github.com/igorbenav/fastcrud/pull/188)

#### What's Changed
* Implement select_schema on EndpointCreator and crud_router by [@ljmc-github](https://github.com/ljmc-github) in [#169](https://github.com/igorbenav/fastcrud/pull/169)
* Custom name bug fix by [@igorbenav](https://github.com/igorbenav) in [#187](https://github.com/igorbenav/fastcrud/pull/187)
* UUID support fix by [@igorbenav](https://github.com/igorbenav) in [#188](https://github.com/igorbenav/fastcrud/pull/188)
* Bump version to 0.15.4 by [@igorbenav](https://github.com/igorbenav) in [#189](https://github.com/igorbenav/fastcrud/pull/189)

#### New Contributors
* [@ljmc-github](https://github.com/ljmc-github) made their first contribution in [#169](https://github.com/igorbenav/fastcrud/pull/169)

**Full Changelog**: https://github.com/igorbenav/fastcrud/compare/v0.15.3...v0.15.4
___
## [0.15.3] - Dec 23, 2024

#### Fixed
- **Get multi joined issue** with repetition and wrong count

#### What's Changed
* Bug fixes by [@igorbenav](https://github.com/igorbenav) in [#186](https://github.com/igorbenav/fastcrud/pull/186)

**Full Changelog**: https://github.com/igorbenav/fastcrud/compare/v0.15.2...v0.15.3
___
## [0.15.2] - Dec 15, 2024

#### Added
- **Add commit option to upsert_multi** by [@feluelle](https://github.com/feluelle) in [#174](https://github.com/igorbenav/fastcrud/pull/174)

#### Fixed
- **Add missing commit to update** by [@feluelle](https://github.com/feluelle) in [#173](https://github.com/igorbenav/fastcrud/pull/173)
- **Default page or items_per_page** for when paginating using the get on list by [@phguyss](https://github.com/phguyss) in [#177](https://github.com/igorbenav/fastcrud/pull/177)
- **Improve update method** when item is not found by [@julianaklulo](https://github.com/julianaklulo) in [#176](https://github.com/igorbenav/fastcrud/pull/176)

#### Improved
- **Fix quick-start documentation** by [@EduardoTT](https://github.com/EduardoTT) in [#178](https://github.com/igorbenav/fastcrud/pull/178)
- **Bump actions to avoid node16 EOL issue** by [@Zatura](https://github.com/Zatura) in [#180](https://github.com/igorbenav/fastcrud/pull/180)
- **Changelog updated** by [@igorbenav](https://github.com/igorbenav) in [#167](https://github.com/igorbenav/fastcrud/pull/167)

#### What's Changed
* Changelog updated by [@igorbenav](https://github.com/igorbenav) in [#167](https://github.com/igorbenav/fastcrud/pull/167)
* Add missing commit to update by [@feluelle](https://github.com/feluelle) in [#173](https://github.com/igorbenav/fastcrud/pull/173)
* Add commit option to upsert_multi by [@feluelle](https://github.com/feluelle) in [#174](https://github.com/igorbenav/fastcrud/pull/174)
* Fix: quick-start documentation by [@EduardoTT](https://github.com/EduardoTT) in [#178](https://github.com/igorbenav/fastcrud/pull/178)
* Improve update method when item is not found by [@julianaklulo](https://github.com/julianaklulo) in [#176](https://github.com/igorbenav/fastcrud/pull/176)
* Bump actions to avoid node16 EOL issue by [@Zatura](https://github.com/Zatura) in [#180](https://github.com/igorbenav/fastcrud/pull/180)
* Fix: default page or items_per_page for when paginating using the get on list by [@phguyss](https://github.com/phguyss) in [#177](https://github.com/igorbenav/fastcrud/pull/177)

#### New Contributors
* [@EduardoTT](https://github.com/EduardoTT) made their first contribution in [#178](https://github.com/igorbenav/fastcrud/pull/178)
* [@julianaklulo](https://github.com/julianaklulo) made their first contribution in [#176](https://github.com/igorbenav/fastcrud/pull/176)
* [@Zatura](https://github.com/Zatura) made their first contribution in [#180](https://github.com/igorbenav/fastcrud/pull/180)
* [@phguyss](https://github.com/phguyss) made their first contribution in [#177](https://github.com/igorbenav/fastcrud/pull/177)

**Full Changelog**: https://github.com/igorbenav/fastcrud/compare/v0.15.1...v0.15.2
___
## [0.15.1] - Sep 18, 2024

#### Added
- **Support for fastapi >=0.100**

#### What's Changed
* now supporting fastapi >= 0.100.0 by @igorbenav in https://github.com/igorbenav/fastcrud/pull/166

**Full Changelog**: https://github.com/igorbenav/fastcrud/compare/v0.15.0...v0.15.1
___
## [0.15.0] - Sep 18, 2024

#### Added
- **Models and Schemas for Task Management (Batch 3)** by [@slaarti](https://github.com/slaarti)
- **Models and Schemas for Articles, Authors, and Profiles (Batch 4)** by [@slaarti](https://github.com/slaarti)
- **`update_override` Argument to `upsert_multi` Method** by [@feluelle](https://github.com/feluelle)
- **Configurable `is_deleted` Field in Soft Delete Logic** by [@gal-dahan](https://github.com/gal-dahan)

#### Improved
- **Fixed Complex Parameter Filter with `between` Operator** by [@wu-clan](https://github.com/wu-clan)
- **Fixed Cryptography Package Vulnerability**
- **Resolved Update Column Name Collision in Update Method**

#### Fixed
- **Vulnerability in `cryptography` Package** updated to `cryptography = "^43.0.1"`
- **Update Column Name Collision** in the `update` method

#### Documentation Updates
- **Added Documentation for New Models and Schemas** by [@slaarti](https://github.com/slaarti)
- **Updated `upsert_multi` Method Documentation with `update_override` Usage** by [@feluelle](https://github.com/feluelle)
- **Clarified Endpoint Simplification and Deprecation Notices**

#### Warnings
- **Deprecation Notice**: The `_read_paginated` endpoint has been removed. Please transition to using `_read_items` with pagination parameters. [Docs here](https://igorbenav.github.io/fastcrud/advanced/endpoint/#read-multiple).
- **Deprecation Notice**: Handling of `Depends` is now only callable within `_inject_depend`. Update your code accordingly.
- **Configuration Change Alert**: Endpoints are simplified by default. Adjust your configurations to align with the new defaults. [Docs here](https://igorbenav.github.io/fastcrud/advanced/endpoint/#available-automatic-endpoints).

___
#### Detailed Changes

##### Endpoint Simplification and Deprecation of `_read_paginated`

###### Description

To streamline API endpoint configurations, endpoints with empty strings as names are now the standard. Additionally, the `_read_paginated` endpoint has been removed, with its functionality merged into `_read_items`.

###### Changes

- **Simplified Endpoint Configuration**: Endpoints can now be defined with empty strings to create cleaner paths.
- **Removed `_read_paginated` Endpoint**: Pagination is now handled via optional parameters in `_read_items`.

###### Usage Examples

**Paginated Read Example:**

```bash
curl -X 'GET' \
  'http://localhost:8000/items?page=2&itemsPerPage=10' \
  -H 'accept: application/json'
```

**Non-Paginated Read Example:**

```bash
curl -X 'GET' \
  'http://localhost:8000/items?offset=0&limit=100' \
  -H 'accept: application/json'

```

##### Warnings

!!! WARNING
    The `_read_paginated` endpoint is deprecated. Use `_read_items` with pagination parameters instead.

!!! WARNING
    Default endpoint names are now empty strings. Adjust your configurations to match the new defaults.

___

##### `update_override` Argument in `upsert_multi` Method

###### Description

The `upsert_multi` method now includes an `update_override` argument, giving developers the ability to override the default update logic during upsert operations. This enhancement provides greater flexibility for custom update scenarios, such as utilizing SQL `CASE` statements or other complex expressions.

###### Changes

- **`update_override` Argument**: Allows custom update logic in `upsert_multi`.
- **Dialect Support**: Implemented for PostgreSQL, SQLite, and MySQL.
- **Tests**: Added comprehensive tests to ensure functionality across different SQL dialects.

###### Usage Example

```python
from fastcrud import FastCRUD
from sqlalchemy import case
from .models.item import Item
from .database import session as db

crud_items = FastCRUD(Item)

await crud_items.upsert_multi(
    db=db,
    instances=[
        ItemCreateSchema(id=1, name="Item A", price=10),
        ItemCreateSchema(id=2, name="Item B", price=20),
    ],
    update_override={
        "price": case(
            (Item.price.is_(None), db.excluded.price),
            else_=Item.price,
        )
    }
)
```

___

##### Configurable `is_deleted` Field in Soft Delete Logic

###### Description

The `is_deleted` field in the soft delete logic is now optional and configurable. This change allows developers to customize the soft delete behavior per model, providing flexibility in how deletion states are handled.

___

#### New Contributors
- [@wu-clan](https://github.com/wu-clan) made their first contribution 🌟
- [@gal-dahan](https://github.com/gal-dahan) made their first contribution 🌟

**Full Changelog**: [View the full changelog](https://github.com/igorbenav/fastcrud/compare/v0.14.0...v0.15.0)
___
## [0.14.0] - Jul 29, 2024

#### Added
- Type-checking support for SQLModel types by @kdcokenny 🚀
- Returning clause to update operations by @feluelle
- Upsert_multi functionality by @feluelle
- Simplified endpoint configurations by @JakNowy, streamlining path generation and merging pagination functionalities into a unified `_read_items` endpoint, promoting more efficient API structure and usage. Details in https://github.com/igorbenav/fastcrud/pull/105

#### Improved
- Comprehensive tests for paginated retrieval of items, maintaining 100% coverage
- Docker client check before running tests that require Docker by @feluelle

#### Fixed
- Vulnerability associated with an outdated cryptography package
- Return type inconsistency in async session fixtures by @slaarti

#### Documentation Updates
- Cleanup of documentation formatting by @slaarti
- Replacement of the Contributing section in docs with an include to file in repo root by @slaarti
- Correction of links to advanced filters in docstrings by @slaarti
- Backfill of docstring fixes across various modules by @slaarti
- Enhanced filter documentation with new AND and OR clause examples, making complex queries more accessible and understandable.

#### Models and Schemas Enhancements
- Introduction of simple and one-off models (Batch 1) by @slaarti
- Expansion to include models and schemas for Customers, Products, and Orders (Batch 2) by @slaarti

#### Code Refinements
- Resolution of missing type specifications in kwargs by @slaarti
- Collapsed space adjustments for models/schemas in `fast_crud.py` by @slaarti

#### Warnings
- **Deprecation Notice**: `_read_paginated` endpoint is set to be deprecated and merged into `_read_items`. Users are encouraged to transition to the latter, utilizing optional pagination parameters. Full details and usage instructions provided to ensure a smooth transition.
- **Future Changes Alert**: Default endpoint names in `EndpointCreator` are anticipated to be set to empty strings in a forthcoming major release, aligning with simplification efforts. Refer to https://github.com/igorbenav/fastcrud/issues/67 for more information.

___
#### Detailed Changes

##### Simplified Endpoint Configurations

In an effort to streamline FastCRUD’s API, we have reconfigured endpoint paths to avoid redundancy (great work by @JakNowy). This change allows developers to specify empty strings for endpoint names in the `crud_router` setup, which prevents the generation of unnecessary `//` in the paths. The following configurations illustrate how endpoints can now be defined more succinctly:

```python
endpoint_names = {
    "create": "",
    "read": "",
    "update": "",
    "delete": "",
    "db_delete": "",
    "read_multi": "",
    "read_paginated": "get_paginated",
}
```

Moreover, the `_read_paginated` logic has been integrated into the `_read_items` endpoint. This integration means that pagination can now be controlled via `page` and `items_per_page` query parameters, offering a unified method for both paginated and non-paginated reads:

- **Paginated read example**:

```bash
curl -X 'GET' \
  'http://localhost:8000/users/get_multi?page=2&itemsPerPage=10' \
  -H 'accept: application/json'
```

- **Non-paginated read example**:

```bash
curl -X 'GET' \
  'http://localhost:8000/users/get_multi?offset=0&limit=100' \
  -H 'accept: application/json'
```

###### Warnings

- **Deprecation Warning**: The `_read_paginated` endpoint is slated for deprecation. Developers should transition to using `_read_items` with the relevant pagination parameters.
- **Configuration Change Alert**: In a future major release, default endpoint names in `EndpointCreator` will be empty strings by default, as discussed in [Issue #67](https://github.com/igorbenav/fastcrud/issues/67).

###### Advanced Filters Documentation Update

Documentation for advanced filters has been expanded to include comprehensive examples of AND and OR clauses, enhancing the utility and accessibility of complex query constructions.

- **OR clause example**:

```python
# Fetch items priced under $5 or above $20
items = await item_crud.get_multi(
    db=db,
    price__or={'lt': 5, 'gt': 20},
)
```

- **AND clause example**:

```python
# Fetch items priced under $20 and over 2 years of warranty
items = await item_crud.get_multi(
    db=db,
    price__lt=20,
    warranty_years__gt=2,
)
```

___
##### Returning Clauses in Update Operations

###### Description
Users can now retrieve updated records immediately following an update operation. This feature streamlines the process, reducing the need for subsequent retrieval calls and increasing efficiency.

###### Changes
- **Return Columns**: Specify the columns to be returned after the update via the `return_columns` argument.
- **Schema Selection**: Optionally select a Pydantic schema to format the returned data using the `schema_to_select` argument.
- **Return as Model**: Decide if the returned data should be converted into a model using the `return_as_model` argument.
- **Single or None**: Utilize the `one_or_none` argument to ensure that either a single record is returned or none, in case the conditions do not match any records.

These additions are aligned with existing CRUD API functions, enhancing consistency across the library and making the new features intuitive for users.

###### Usage Example

###### Returning Updated Fields

```python
from fastcrud import FastCRUD
from .models.item import Item
from .database import session as db

crud_items = FastCRUD(Item)
updated_item = await crud_items.update(
    db=db,
    object={"price": 9.99},
    price__lt=10,
    return_columns=["price"]
)
# This returns the updated price of the item directly.
```

###### Returning Data as a Model

```python
from fastcrud import FastCRUD
from .models.item import Item
from .schemas.item import ItemSchema
from .database import session as db

crud_items = FastCRUD(Item)
updated_item_schema = await crud_items.update(
    db=db,
    object={"price": 9.99},
    price__lt=10,
    schema_to_select=ItemSchema,
    return_as_model=True
)
# This returns the updated item data formatted as an ItemSchema model.
```

___
##### Bulk Upsert Operations with `upsert_multi`

The `upsert_multi` method provides the ability to perform bulk upsert operations, which are optimized for different SQL dialects.

###### Changes
- **Dialect-Optimized SQL**: Uses the most efficient SQL commands based on the database's SQL dialect.
- **Support for Multiple Dialects**: Includes custom implementations for PostgreSQL, SQLite, and MySQL, with appropriate handling for each's capabilities and limitations.

###### Usage Example

###### Upserting Multiple Records

```python
from fastcrud import FastCRUD
from .models.item import Item
from .schemas.item import ItemCreateSchema, ItemSchema
from .database import session as db

crud_items = FastCRUD(Item)
items = await crud_items.upsert_multi(
    db=db,
    instances=[
        ItemCreateSchema(price=9.99),
    ],
    schema_to_select=ItemSchema,
    return_as_model=True,
)
# This will return the upserted data in the form of ItemSchema.
```

###### Implementation Details

`upsert_multi` handles different database dialects:
- **PostgreSQL**: Uses `ON CONFLICT DO UPDATE`.
- **SQLite**: Utilizes `ON CONFLICT DO UPDATE`.
- **MySQL**: Implements `ON DUPLICATE KEY UPDATE`.

###### Notes
- MySQL and MariaDB do not support certain advanced features used in other dialects, such as returning values directly after an insert or update operation. This limitation is clearly documented to prevent misuse.

#### New Contributors
- @kdcokenny made their first contribution 🌟
- @feluelle made their first contribution 🌟

**Full Changelog**: [View the full changelog](https://github.com/igorbenav/fastcrud/compare/v0.13.1...v0.14.0)


## [0.13.1] - Jun 22, 2024

#### Added
- More Advanced Filters by @JakNowy 🎉

#### Fixed
- Bug where objects with null primary key are returned with all fields set to None in nested joins #102

___
#### Detailed Changes

### Advanced Filters

FastCRUD supports advanced filtering options, allowing you to query records using operators such as greater than (`__gt`), less than (`__lt`), and their inclusive counterparts (`__gte`, `__lte`). These filters can be used in any method that retrieves or operates on records, including `get`, `get_multi`, `exists`, `count`, `update`, and `delete`.

#### Single parameter filters

Most filter operators require a single string or integer value.

```python
# Fetch items priced above $5
items = await item_crud.get_multi(
    db=db,
    price__gte=5,
)
```

Currently supported single parameter filters are:
- __gt - greater than
- __lt - less than
- __gte - greater than or equal to
- __lte - less than or equal to
- __ne - not equal
- __is - used to test True, False, and None identity
- __is_not - negation of "is"
- __like - SQL "like" search for specific text pattern
- __notlike - negation of "like"
- __ilike - case insensitive "like"
- __notilike - case insensitive "notlike"
- __startswith - text starts with given string
- __endswith - text ends with given string
- __contains - text contains given string
- __match - database-specific match expression

#### Complex parameter filters

Some operators require multiple values. They must be passed as a python tuple, list, or set.

```python
# Fetch items priced between $5 and $20
items = await item_crud.get_multi(
    db=db,
    price__between=(5, 20),
)
```
- __between - between 2 numeric values
- __in - included in
- __not_in - not included in

#### OR parameter filters

More complex OR filters are supported. They must be passed as a dictionary, where each key is a library-supported operator to be used in OR expression and values are what get's passed as the parameter.

```python
# Fetch items priced under $5 or above $20
items = await item_crud.get_multi(
    db=db,
    price__or={'lt': 5, 'gt': 20},
)
```

#### What's Changed
- Missing sqlalchemy operators by [@JakNowy](https://github.com/JakNowy) in https://github.com/igorbenav/fastcrud/pull/85
- Null primary key bug fixed in https://github.com/igorbenav/fastcrud/pull/107

**Full Changelog**: https://github.com/igorbenav/fastcrud/compare/v0.13.0...v0.13.1

## [0.13.0] - May 28, 2024

#### Added
- Filters in Automatic Endpoints 🎉
- One-to-many support in joins
- Upsert method in FastCRUD class by @dubusster

___
#### Detailed Changes

##### Using Filters in FastCRUD

FastCRUD provides filtering capabilities, allowing you to filter query results based on various conditions. Filters can be applied to `read_multi` and `read_paginated` endpoints. This section explains how to configure and use filters in FastCRUD.


##### Defining Filters

Filters are either defined using the `FilterConfig` class or just passed as a dictionary. This class allows you to specify default filter values and validate filter types. Here's an example of how to define filters for a model:

```python
from fastcrud import FilterConfig

# Define filter configuration for a model
filter_config = FilterConfig(
    tier_id=None,  # Default filter value for tier_id
    name=None  # Default filter value for name
)
```

And the same thing using a `dict`:
```python
filter_config = {
    "tier_id": None,  # Default filter value for tier_id
    "name": None,  # Default filter value for name
}
```

By using `FilterConfig` you get better error messages.

###### Applying Filters to Endpoints

You can apply filters to your endpoints by passing the `filter_config` to the `crud_router` or `EndpointCreator`. Here's an example:

```python
from fastcrud import crud_router
from yourapp.models import YourModel
from yourapp.schemas import CreateYourModelSchema, UpdateYourModelSchema
from yourapp.database import async_session

# Apply filters using crud_router
app.include_router(
    crud_router(
        session=async_session,
        model=YourModel,
        create_schema=CreateYourModelSchema,
        update_schema=UpdateYourModelSchema,
        filter_config=filter_config,  # Apply the filter configuration
        path="/yourmodel",
        tags=["YourModel"]
    )
)
```

###### Using Filters in Requests

Once filters are configured, you can use them in your API requests. Filters are passed as query parameters. Here's an example of how to use filters in a request to a paginated endpoint:

```http
GET /yourmodel/get_paginated?page=1&itemsPerPage=3&tier_id=1&name=Alice
```

###### Custom Filter Validation

The `FilterConfig` class includes a validator to check filter types. If an invalid filter type is provided, a `ValueError` is raised. You can customize the validation logic by extending the `FilterConfig` class:

```python
from fastcrud import FilterConfig
from pydantic import ValidationError

class CustomFilterConfig(FilterConfig):
    @field_validator("filters")
    def check_filter_types(cls, filters: dict[str, Any]) -> dict[str, Any]:
        for key, value in filters.items():
            if not isinstance(value, (type(None), str, int, float, bool)):
                raise ValueError(f"Invalid default value for '{key}': {value}")
        return filters

try:
    # Example of invalid filter configuration
    invalid_filter_config = CustomFilterConfig(invalid_field=[])
except ValidationError as e:
    print(e)
```

###### Handling Invalid Filter Columns

FastCRUD ensures that filters are applied only to valid columns in your model. If an invalid filter column is specified, a `ValueError` is raised:

```python
try:
    # Example of invalid filter column
    invalid_filter_config = FilterConfig(non_existent_column=None)
except ValueError as e:
    print(e)  # Output: Invalid filter column 'non_existent_column': not found in model
```

___
##### Handling One-to-One and One-to-Many Joins in FastCRUD

FastCRUD provides flexibility in handling one-to-one and one-to-many relationships through `get_joined` and `get_multi_joined` methods, along with the ability to specify how joined data should be structured using both the `relationship_type` (default `one-to-one`) and the `nest_joins` (default `False`) parameters.

###### One-to-One Relationships
- **`get_joined`**: Fetch a single record and its directly associated record (e.g., a user and their profile).
- **`get_multi_joined`** (with `nest_joins=False`): Retrieve multiple records, each linked to a single related record from another table (e.g., users and their profiles).


**Example**

Let's define two tables:

```python
class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    tier_id = Column(Integer, ForeignKey("tier.id"))

class Tier(Base):
    __tablename__ = "tier"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
```

Fetch a user and their tier:

```python
user_tier = await user_crud.get_joined(
    db=db,
    join_model=Tier,
    join_on=User.tier_id == Tier.id,
    join_type="left",
    join_prefix="tier_",
    id=1
)
```

The result will be:

```json
{
    "id": 1,
    "name": "Example",
    "tier_id": 1,
    "tier_name": "Free"
}
```

**One-to-One Relationship with Nested Joins**

To get the joined data in a nested dictionary:

```python
user_tier = await user_crud.get_joined(
    db=db,
    join_model=Tier,
    join_on=User.tier_id == Tier.id,
    join_type="left",
    join_prefix="tier_",
    nest_joins=True,
    id=1
)
```

The result will be:

```json
{
    "id": 1,
    "name": "Example",
    "tier": {
        "id": 1,
        "name": "Free"
    }
}
```


###### One-to-Many Relationships
- **`get_joined`** (with `nest_joins=True`): Retrieve a single record with all its related records nested within it (e.g., a user and all their blog posts).
- **`get_multi_joined`** (with `nest_joins=True`): Fetch multiple primary records, each with their related records nested (e.g., multiple users and all their blog posts).

!!! WARNING
    When using `nest_joins=True`, the performance will always be a bit worse than when using `nest_joins=False`. For cases where more performance is necessary, consider using `nest_joins=False` and remodeling your database.


**Example**

To demonstrate a one-to-many relationship, let's assume `User` and `Post` tables:

```python
class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary key=True)
    name = Column(String)

class Post(Base):
    __tablename__ = "post"
    id = Column(Integer, primary key=True)
    user_id = Column(Integer, ForeignKey("user.id"))
    content = Column(String)
```

Fetch a user and all their posts:

```python
user_posts = await user_crud.get_joined(
    db=db,
    join_model=Post,
    join_on=User.id == Post.user_id,
    join_type="left",
    join_prefix="post_",
    nest_joins=True,
    id=1
)
```

The result will be:

```json
{
    "id": 1,
    "name": "Example User",
    "posts": [
        {
            "id": 101,
            "user_id": 1,
            "content": "First post content"
        },
        {
            "id": 102,
            "user_id": 1,
            "content": "Second post content"
        }
    ]
}
```


## What's Changed
- feat: ✨ add upsert method in FastCRUD class by [@dubusster](https://github.com/dubusster)
- Filters in Automatic Endpoints
- One-to-many support in joins
- tests fixed by @igorbenav
- Using the same session for all tests
- warning added to docs


**Full Changelog**: https://github.com/igorbenav/fastcrud/compare/v0.12.1...v0.13.0

## [0.12.1] - May 10, 2024

#### Added
- Deprecation Warning for dependency handling.

___
#### Detailed Changes

If you pass a sequence of `params.Depends` type variables to any `*_deps` parameter in `EndpointCreator` and `crud_router`, you'll get a warning. Support will be completely removed in 0.15.0.

**Full Changelog**: https://github.com/igorbenav/fastcrud/compare/v0.12.0...v0.12.1


## [0.12.0] - May 8, 2024

#### Added
- Unpaginated versions of multi-row get methods by @slaarti in #62  🎉
- Nested Join bug fixes
- Dependency handling now working as docs say
- Option to Skip commit in some fastcrud methods
- Docstring example fixes
- `__in` and `__not_in` filters by @JakNowy 🎉
- Fastapi 0.111.0 support

___
#### Detailed Changes

##### Unpaginated versions of multi-row get methods
Now, if you pass `None` to `limit` in `get_multi` and `get_multi_joined`, you get the whole unpaginated set of data that matches the filters. Use this with caution.

```python
from fastcrud import FastCRUD
from .models.item import Item
from .database import session as db

crud_items = FastCRUD(Item)
items = await crud_items.get_multi(db=db, limit=None)
# this will return all items in the db
```

##### Dependency handling now working as docs say
Now, you may pass dependencies to `crud_router` or `EndpointCreator` as simple functions instead of needing to wrap them in `fastapi.Depends`.

```python
from .dependencies import get_superuser
app.include_router(
    crud_router(
        session=db,
        model=Item,
        create_schema=ItemCreate,
        update_schema=ItemUpdate,
        delete_schema=ItemDelete,
        create_deps=[get_superuser],
        update_deps=[get_superuser],
        delete_deps=[get_superuser],
        path="/item",
        tags=["item"],
    )
)
```

##### Option to Skip commit in some fastcrud methods
For `create`, `update`, `db_delete` and `delete` methods of `FastCRUD`, now you have the option of passing `commit=False` so you don't commit the operations immediately.

```python
from fastcrud import FastCRUD
from .models.item import Item
from .database import session as db

crud_items = FastCRUD(Item)

await crud_items.delete(
    db=db,
    commit=False,
    id=1
)
# this will not actually delete until you run a db.commit()
```

##### `__in` and `__not_in` filters
You may now pass `__in` and `__not_in` to methods that accept advanced queries:

- `__gt`: greater than,
- `__lt`: less than,
- `__gte`: greater than or equal to,
- `__lte`: less than or equal to,
- `__ne`: not equal,
- `__in`: included in [tuple, list or set],
- `__not_in`: not included in [tuple, list or set].

#### What's Changed
- Add unpaginated versions of multi-row get methods (w/tests) by [@slaarti](https://github.com/slaarti) 🎉
- Join fixes
- Dependencies
- Skip commit
- Docstring fix
- feat: filter __in by [@JakNowy](https://github.com/JakNowy) 🎉
- python support for 0.111.0 added
- version bump in pyproject.toml for 0.12.0

#### New Contributors
* [@slaarti](https://github.com/slaarti) made their first contribution in https://github.com/igorbenav/fastcrud/pull/62 🎉

**Full Changelog**: https://github.com/igorbenav/fastcrud/compare/v0.11.1...v0.12.0


## [0.11.1] - Apr 22, 2024

#### Added
- `one_or_none` parameter to FastCRUD `get` method (default `False`)
- `nest_joins` parameter to FastCRUD `get_joined` and `get_multi_joined` (default `False`)

___
#### Detailed Changes

##### `get`
By default, the `get` method in `FastCRUD` returns the `first` object matching all the filters it finds.

If you want to ensure the `one_or_none` behavior, you may pass the parameter as `True`:

```python
crud.get(
    async_session,
    one_or_none=True,
    category_id=1
)
```

##### `get_joined` and `get_multi_joined`
By default, `FastCRUD` joins all the data and returns it in a single dictionary.
Let's define two tables:
```python
class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    tier_id = Column(Integer, ForeignKey("tier.id"))


class Tier(Base):
    __tablename__ = "tier"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
```

And join them with `FastCRUD`:

```python
user_tier = await user_crud.get_joined(
    db=db,
    model=Tier,
    join_on=User.tier_id == Tier.id,
    join_type="left",
    join_prefix="tier_",,
    id=1
)
```

We'll get:

```javascript
{
    "id": 1,
    "name": "Example",
    "tier_id": 1,
    "tier_name": "Free",
}
```

Now, if you want the joined data in a nested dictionary instead, you may just pass `nest_joins=True`:

```python
user_tier = await user_crud.get_joined(
    db=db,
    model=Tier,
    join_on=User.tier_id == Tier.id,
    join_type="left",
    join_prefix="tier_",
    nest_joins=True,
    id=1,
)
```

And you will get:

```javascript
{
    "id": 1,
    "name": "Example",
    "tier": {
        "id": 1,
        "name": "Free",
    },
}
```

This works for both `get_joined` and `get_multi_joined`.

!!! WARNING
    Note that the final `"_"` in the passed `"tier_"` is stripped.

#### What's Changed
- Reuse of `select` method in `FastCRUD`
- Skip count call when possible
- Add `one_or_none` parameter to FastCRUD `get` method
- Add `nest_joins` parameter to FastCRUD `get_joined` and `get_multi_joined`

#### New Contributors
- [@JakNowy](https://github.com/JakNowy) made their first contribution in PR #51.

**Full Changelog**: [View the full changelog](https://github.com/igorbenav/fastcrud/compare/v0.11.0...v0.11.1)

## [0.11.0] - Apr 7, 2024

#### Added
- Multiple primary keys support, a significant enhancement by @dubusster in #31 🎉.
- Option to disable the count in `get_multi` and `get_multi_joined` methods for performance optimization.
- Fixes for a validation bug when `return_as_model` is set to `True`.
- Resolution of a bug concerning incorrect handling of `db_row` in methods.
- Correction of the `valid_methods` bug, which previously raised the wrong error type.
- Upgrade of `FastAPI` dependency to version `0.111.0`, ensuring compatibility with the latest FastAPI features.
- Achievement of 100% test coverage, with the addition of a workflow and badge to showcase this milestone.
- Inclusion of the changelog within the project documentation, providing a comprehensive history of changes directly to users.

___
#### Detailed Changes

##### Multiple Primary Keys Support
FastCRUD now accommodates models with multiple primary keys, facilitating more complex database designs. For models defined with more than one primary key, the endpoint creator automatically generates paths reflecting the primary keys' order. This feature extends support to primary keys named differently than `id`, enhancing the flexibility of FastCRUD's routing capabilities.

###### Example:
For a model with multiple primary keys, FastCRUD generates specific endpoints such as `/multi_pk/get/{id}/{uuid}`, accommodating the unique identification needs of complex data models.

##### Optional Count
The `get_multi` and `get_multi_joined` methods now feature an `return_total_count=False` parameter, allowing users to opt-out of receiving the total count in responses. This option can enhance performance by skipping potentially expensive count operations.

###### Behavior:
- By default, `return_total_count=True` is assumed, returning both data and a total count.
- When set to `False`, responses contain only the data array, omitting the total count for efficiency.

#### What's Changed
- Implementation of multiple primary keys support, addressing a significant flexibility requirement for advanced use cases.
- Introduction of optional count retrieval in multi-get methods, optimizing performance by eliminating unnecessary database queries.
- Several critical bug fixes, improving the stability and reliability of FastCRUD.
- Documentation enhancements, including the addition of a changelog section, ensuring users have access to detailed release information.
- Update to FastAPI `0.111.0`, ensuring compatibility with the latest enhancements in the FastAPI ecosystem.
- Achievement of 100% test coverage, marking a significant milestone in the project's commitment to reliability and quality assurance.

#### Relevant Contributors
- [@dubusster](https://github.com/dubusster) made a notable contribution with the implementation of multiple primary keys support in PR #31.

**Full Changelog**: [View the full changelog](https://github.com/igorbenav/fastcrud/compare/v0.10.0...v0.11.0)

## [0.10.0] - Mar 30, 2024

#### Added
- `select` statement functionality, thanks to @dubusster's contribution in PR #28 🚀.
- Support for joined models in the `count` method through the `joins_config` parameter.
- Filters for joined models via the `filters` parameter in `JoinConfig`.
- Type checking workflow integrated with `mypy` alongside fixes for typing issues.
- Linting workflow established with `ruff`.

___
#### Detailed Changes

##### Select
The `select` method constructs a SQL Alchemy `Select` statement, offering flexibility in column selection, filtering, and sorting. It is designed to chain additional SQLAlchemy methods for complex queries.
[Docs here](https://igorbenav.github.io/fastcrud/usage/crud/#5-select) and [here](https://igorbenav.github.io/fastcrud/advanced/crud/#the-select-method).

###### Features:
- **Column Selection**: Choose columns via a Pydantic schema.
- **Sorting**: Define columns and their order for sorting.
- **Filtering**: Directly apply filters through keyword arguments.
- **Chaining**: Allow for chaining with other SQLAlchemy methods for advanced query construction.

##### Improved Joins
`JoinConfig` enhances FastCRUD queries by detailing join operations between models, offering configurations like model joining, conditions, prefixes, column selection through schemas, join types, aliases, and direct filtering. [Docs here](https://igorbenav.github.io/fastcrud/advanced/joins/).

#### Applying Joins in FastCRUD Methods
Detailed explanations and examples are provided for using joins in `count`, `get_joined`, and `get_multi_joined` methods to achieve complex data retrieval, including handling of many-to-many relationships.

#### What's Changed
- New `select` statement functionality added.
- Documentation and method improvements for select and joins.
- Integration of type checking and linting workflows.
- Version bump in pyproject.toml.

#### New Contributors
- [@dubusster](https://github.com/dubusster) made their first contribution in PR #28.

**Full Changelog**: [View the full changelog](https://github.com/igorbenav/fastcrud/compare/v0.9.1...v0.10.0)

___

## [0.9.1] - Mar 19, 2024

#### Added
- Enhanced `get_joined` and `get_multi_joined` methods to support aliases, enabling multiple joins on the same model. This improvement addresses issue #27.

___
#### Detailed Changes

##### Alias Support for Complex Joins
With the introduction of alias support, `get_joined` and `get_multi_joined` methods now allow for more complex queries, particularly beneficial in scenarios requiring self-joins or multiple joins on the same table. Aliases help to avoid conflicts and ambiguity by providing unique identifiers for the same model in different join contexts. [Docs here](https://igorbenav.github.io/fastcrud/advanced/joins/#complex-joins-using-joinconfig).

###### Example: Multiple Joins with Aliases
To demonstrate the use of aliases, consider a task management system where tasks are associated with both an owner and an assigned user from the same `UserModel`. Aliases enable joining the `UserModel` twice under different contexts - as an owner and an assigned user. This example showcases how to set up aliases using the `aliased` function and incorporate them into your `JoinConfig` for clear and conflict-free query construction. [Docs here](https://igorbenav.github.io/fastcrud/advanced/crud/#example-joining-the-same-model-multiple-times).

#### What's Changed
- Introduction of aliases in joins, improving query flexibility and expressiveness, as detailed by @igorbenav in PR #29.

**Full Changelog**: [View the full changelog](https://github.com/igorbenav/fastcrud/compare/v0.9.0...v0.9.1)

___

## [0.9.0] - Mar 14, 2024

#### Added
- Enhanced `get_joined` and `get_multi_joined` methods now support handling joins with multiple models.

___
#### Detailed Changes

##### Multi-Model Join Capabilities
The `get_joined` and `get_multi_joined` methods have been upgraded to accommodate joins involving multiple models. This functionality is facilitated through the `joins_config` parameter, allowing for the specification of multiple `JoinConfig` instances. Each instance represents a unique join configuration, broadening the scope for complex data relationship management within FastCRUD. [Docs here](https://igorbenav.github.io/fastcrud/advanced/joins/).

###### Example: Multi-Model Join
A practical example involves retrieving users alongside their corresponding tier and department details. By configuring `joins_config` with appropriate `JoinConfig` instances for the `Tier` and `Department` models, users can efficiently gather comprehensive data across related models, enhancing data retrieval operations' depth and flexibility.

!!! WARNING
    An error will occur if both single join parameters and `joins_config` are used simultaneously. It's crucial to ensure that your join configurations are correctly set to avoid conflicts.

#### What's Changed
- Introduction of multi-model join support in `get_joined` and `get_multi_joined`, enabling more complex and detailed data retrieval strategies.
- Several minor updates and fixes, including package import corrections and `pyproject.toml` updates, to improve the library's usability and stability.

#### New Contributors
- [@iridescentGray](https://github.com/iridescentGray)

**Full Changelog**: [View the full changelog](https://github.com/igorbenav/fastcrud/compare/v0.8.0...v0.9.0)

___

## [0.8.0] - Mar 4, 2024

#### Added
- Feature to customize names of auto-generated endpoints using the `endpoint_names` parameter, applicable in both `crud_router` function and `EndpointCreator`.

___
#### Detailed Changes

##### Custom Endpoint Naming
The introduction of the `endpoint_names` parameter offers flexibility in defining endpoint names for CRUD operations. This enhancement caters to the need for more descriptive or project-specific naming conventions, enabling developers to align the API's interface with their domain language or organizational standards.

###### Example: Customizing Endpoint Names with `crud_router`
Customizing endpoint names is straightforward with the `crud_router` function. By providing a dictionary mapping CRUD operation names to desired endpoint names, developers can easily tailor their API's paths to fit their application's unique requirements.

###### Example: Customizing Endpoint Names with `EndpointCreator`
Similarly, when using the `EndpointCreator`, the `endpoint_names` parameter allows for the same level of customization, ensuring consistency across different parts of the application or service.

!!! TIP

    It's not necessary to specify all endpoint names; only those you wish to change need to be included in the `endpoint_names` dictionary. This flexibility ensures minimal configuration effort for maximum impact.

#### What's Changed
- Enhanced endpoint customization capabilities through `endpoint_names` parameter, supporting a more tailored and intuitive API design.
- Documentation updates to guide users through the process of customizing endpoint names.

___

## [0.7.0] - Feb 20, 2024

#### Added
- The `get_paginated` endpoint for retrieving items with pagination support.
- The `paginated` module to offer utility functions for pagination.

___
#### Detailed Changes

##### `get_paginated` Endpoint
This new endpoint enhances data retrieval capabilities by introducing pagination, an essential feature for handling large datasets efficiently. It supports customizable query parameters for page number and items per page, facilitating flexible data access patterns. [Docs here](https://igorbenav.github.io/fastcrud/advanced/endpoint/#read-paginated).

###### Features:
- **Endpoint and Method**: A `GET` request to `/get_paginated`.
- **Query Parameters**: Includes `page` for the page number and `itemsPerPage` for controlling the number of items per page.
- **Example Usage**: Demonstrated with a request for retrieving items with specified pagination settings.

##### `paginated` Module
The introduction of the `paginated` module brings two key utility functions, `paginated_response` and `compute_offset`, which streamline the implementation of paginated responses in the application.

###### Functions:
- **paginated_response**: Constructs a paginated response based on the input data, page number, and items per page.
- **compute_offset**: Calculates the offset for database queries, based on the current page number and the number of items per page.

#### What's Changed
- Deployment of pagination functionality, embodied in the `get_paginated` endpoint and the `paginated` module, to facilitate efficient data handling and retrieval.

**Full Changelog**: [View the full changelog](https://github.com/igorbenav/fastcrud/compare/v0.6.0...v0.7.0)

___

## [0.6.0] - Feb 11, 2024

#### Added
- The ability to use a custom `updated_at` column name in models.
- Making the passing of the `crud` parameter to `crud_router` and `EndpointCreator` optional.
- Inclusion of exceptions in the `http_exceptions` module within the broader `exceptions` module for better organization and accessibility.

#### Detailed Changes

##### Custom `updated_at` Column
FastCRUD now supports the customization of the `updated_at` column name, providing flexibility for applications with different database schema conventions or naming practices. [Docs here](https://igorbenav.github.io/fastcrud/advanced/endpoint/#using-endpointcreator-and-crud_router-with-custom-soft-delete-or-update-columns).

###### Example Configuration:
The example demonstrates how to specify a custom column name for `updated_at` when setting up the router for an endpoint, allowing for seamless integration with existing database schemas.

#### What's Changed
- Introduction of features enhancing flexibility and usability, such as custom `updated_at` column names and the optional CRUD parameter in routing configurations.

**Full Changelog**: [View the full changelog](https://github.com/igorbenav/fastcrud/compare/v0.5.0...v0.6.0)

___

## [0.5.0] - Feb 3, 2024

#### Added
- Advanced filters inspired by Django ORM for enhanced querying capabilities.
- Optional bulk operations for update and delete methods.
- Custom soft delete mechanisms integrated into `FastCRUD`, `EndpointCreator`, and `crud_router`.
- Comprehensive test suite for the newly introduced features.

___
#### Detailed Changes

##### Advanced Filters
The advanced filtering system allows for sophisticated querying with support for operators like `__gt`, `__lt`, `__gte`, and `__lte`, applicable across various CRUD operations. This feature significantly enhances the flexibility and power of data retrieval and manipulation within FastCRUD. [Docs here](https://igorbenav.github.io/fastcrud/advanced/crud/#advanced-filters).

###### Examples:
- Utilization of advanced filters for precise data fetching and aggregation.
- Implementation examples for fetching records within specific criteria and counting records based on date ranges.

##### Custom Soft Delete Mechanisms
FastCRUD's soft delete functionality now supports customization, allowing developers to specify alternative column names for `is_deleted` and `deleted_at` fields. This adaptation enables seamless integration with existing database schemas that employ different naming conventions for soft deletion tracking. [Docs here](https://igorbenav.github.io/fastcrud/advanced/endpoint/#using-endpointcreator-and-crud_router-with-custom-soft-delete-or-update-columns).

###### Example Configuration:
- Setting up `crud_router` with custom soft delete column names, demonstrating the flexibility in adapting FastCRUD to various database schema requirements.

##### Bulk Operations
The introduction of optional bulk operations for updating and deleting records provides a more efficient way to handle large datasets, enabling mass modifications or removals with single method calls. This feature is particularly useful for applications that require frequent bulk data management tasks. [Docs here](https://igorbenav.github.io/fastcrud/advanced/crud/#allow-multiple-updates-and-deletes).

###### Examples:
- Demonstrating bulk update and delete operations, highlighting the capability to apply changes to multiple records based on specific criteria.

#### What's Changed
- Addition of advanced filters, bulk operations, and custom soft delete functionalities.

#### New Contributors
- [@YuriiMotov](https://github.com/YuriiMotov)

**Full Changelog**: [View the full changelog](https://github.com/igorbenav/fastcrud/compare/v0.4.0...v0.5.0)

___

## [0.4.0] - Jan 31, 2024

#### Added
- Documentation and tests for SQLModel support.
- `py.typed` file for better typing support.

#### Detailed

Check the [docs for SQLModel support](https://igorbenav.github.io/fastcrud/sqlmodel/).

#### What's Changed
- SQLModel support.

**Full Changelog**: [View the full changelog](https://github.com/igorbenav/fastcrud/compare/v0.3.0...v0.4.0)

___

## [0.3.0] - Jan 28, 2024

#### Added
- The `CustomEndpointCreator` for advanced route creation and customization.
- The ability to selectively include or exclude CRUD operations in the `crud_router` using `included_methods` and `deleted_methods`.
- Comprehensive tests for the new features.
- Detailed documentation on utilizing the `CustomEndpointCreator` and selectively including or excluding endpoints.

#### CustomEndpointCreator
This feature introduces the capability to extend the `EndpointCreator` class, enabling developers to define custom routes and incorporate complex logic into API endpoints. The documentation has been updated to include detailed examples and guidelines on implementing and using `CustomEndpointCreator` in projects. [Docs here](https://igorbenav.github.io/fastcrud/advanced/endpoint/#creating-a-custom-endpointcreator).

#### Selective CRUD Operations
The `crud_router` function has been enhanced with `included_methods` and `deleted_methods` parameters, offering developers precise control over which CRUD methods are included or excluded when configuring routers. This addition provides flexibility in API design, allowing for the creation of tailored endpoint setups that meet specific project requirements. [Docs here](https://igorbenav.github.io/fastcrud/advanced/endpoint/#selective-crud-operations).

___
#### Detailed Changes

#### Extending EndpointCreator
Developers can now create a subclass of `EndpointCreator` to define custom routes or override existing methods, adding a layer of flexibility and customization to FastCRUD's routing capabilities.

##### Creating a Custom EndpointCreator
An example demonstrates how to subclass `EndpointCreator` and add custom routes or override existing methods, further illustrating how to incorporate custom endpoint logic and route configurations into the FastAPI application.

##### Adding Custom Routes
The process involves overriding the `add_routes_to_router` method to include both standard CRUD routes and custom routes, showcasing how developers can extend FastCRUD's functionality to suit their application's unique needs.

##### Using the Custom EndpointCreator
An example highlights how to use the custom `EndpointCreator` with `crud_router`, specifying selective methods to be included in the router setup, thereby demonstrating the practical application of custom endpoint creation and selective method inclusion.

##### Selective CRUD Operations
Examples for using `included_methods` and `deleted_methods` illustrate how to specify exactly which CRUD methods should be included or excluded when setting up the router, offering developers precise control over their API's exposed functionality.

!!! WARNING
    Providing both `included_methods` and `deleted_methods` will result in a ValueError.

**Full Changelog**: [View the full changelog](https://github.com/igorbenav/fastcrud/compare/v0.2.1...v0.3.0)

___

## [0.2.1] - Jan 27, 2024

### What's Changed
- Improved type hints across the codebase, enhancing the clarity and reliability of type checking within FastCRUD.
- Documentation has been thoroughly updated and refined, including fixes for previous inaccuracies and the addition of more detailed explanations and examples.
- Descriptions have been added to automatically generated endpoints, making the API documentation more informative and user-friendly.

**Full Changelog**: [View the full changelog](https://github.com/igorbenav/fastcrud/compare/v0.2.0...v0.2.1)

___

## [0.2.0] - Jan 25, 2024

### Added
- [Docs Published!](https://igorbenav.github.io/fastcrud/)

___

## [0.1.5] - Jan 24, 2024

Readme updates, pyproject requirements

___

## [0.1.2] - Jan 23, 2024

First public release.
