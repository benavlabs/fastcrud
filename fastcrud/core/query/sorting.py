"""
SQL Sorting Processor.

This module handles SQL ORDER BY clause generation with support for
single and multi-column sorting with customizable sort directions.
"""

from sqlalchemy import Select, asc, desc
from sqlalchemy.exc import ArgumentError

from ..introspection import get_model_column


class SortProcessor:
    """Handles SQL ORDER BY clause generation."""

    def __init__(self, model: type):
        """
        Initialize sort processor for a specific model.

        Args:
            model: SQLAlchemy model class
        """
        self.model = model

    def apply_sorting_to_statement(
        self,
        stmt: Select,
        sort_columns: str | list[str],
        sort_orders: str | list[str] | None = None,
    ) -> Select:
        """
        Apply sorting to a SQLAlchemy SELECT statement.

        This method provides flexible sorting capabilities supporting both
        single and multi-column sorting with customizable sort directions.

        Args:
            stmt: SQLAlchemy SELECT statement to modify
            sort_columns: Column name(s) to sort by
            sort_orders: Sort direction(s) - 'asc' or 'desc'

        Returns:
            Modified SELECT statement with ORDER BY clause

        Raises:
            ValueError: If sort parameters are invalid
            ArgumentError: If sort column doesn't exist in model

        Example:
            >>> processor = SortProcessor(User)
            >>> stmt = select(User)
            >>> sorted_stmt = processor.apply_sorting_to_statement(
            ...     stmt, ['name', 'created_at'], ['asc', 'desc']
            ... )
        """
        if sort_orders and not sort_columns:
            raise ValueError("Sort orders provided without corresponding sort columns.")

        if not sort_columns:
            return stmt

        if isinstance(sort_columns, str):
            sort_columns = [sort_columns]

        if sort_orders is None:
            sort_orders = ["asc"] * len(sort_columns)
        elif isinstance(sort_orders, str):
            sort_orders = [sort_orders]

        if len(sort_columns) != len(sort_orders):
            raise ValueError(
                f"Length of sort_columns ({len(sort_columns)}) must match "
                f"length of sort_orders ({len(sort_orders)})"
            )

        order_clauses = []
        for column_name, order in zip(sort_columns, sort_orders):
            if order.lower() not in ("asc", "desc"):
                raise ValueError(
                    f"Invalid sort order: {order}. Must be 'asc' or 'desc'"
                )

            try:
                column = get_model_column(self.model, column_name)

                if order.lower() == "desc":
                    order_clauses.append(desc(column))
                else:
                    order_clauses.append(asc(column))

            except ValueError as e:
                raise ArgumentError(f"Invalid sort column '{column_name}': {e}")

        return stmt.order_by(*order_clauses)
