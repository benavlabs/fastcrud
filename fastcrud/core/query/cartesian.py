"""
Refusing statements that read tables they never join.

A condition naming a model a statement doesn't join adds that model's table to
the FROM clause, and every row is paired with every row of it. SQLAlchemy only
warns about that cartesian product; these helpers refuse it before it runs.
"""

from typing import TypeGuard

from sqlalchemy import Select
from sqlalchemy.sql.elements import _anonymous_label
from sqlalchemy.sql.selectable import FromClause, Join


def refuse_a_cartesian_product(statement: Select) -> None:
    """
    Refuse a statement that reads tables it never joins.

    A condition naming a model the query doesn't join adds that model's table to
    the FROM clause on its own, and every row is paired with every row of it.
    Subqueries such as ``Author.articles.any(...)`` don't, since they're
    correlated with the query.

    Args:
        statement: The SELECT statement about to run, or one with the same FROM
            clause and conditions as an update or delete

    Raises:
        ValueError: Naming the tables read side by side

    Example:
        >>> refuse_a_cartesian_product(select(Author).where(Article.title == "x"))
        Traceback (most recent call last):
        ValueError: The query reads author and article side by side without joining them...
    """
    froms = statement.get_final_froms()
    if len(froms) > 1:
        names = [describe_from_clause(f) for f in froms]
        raise ValueError(
            f"The query reads {', '.join(names[:-1])} and {names[-1]} side by side "
            "without joining them, which pairs every row of each with every row of "
            "the others. A condition names a model the query doesn't join: join it, "
            "name its alias if the join uses one, or test it in a subquery such as "
            "Author.articles.any(...)."
        )


def describe_from_clause(from_clause: FromClause) -> str:
    """Name a FROM element by its tables: ``author`` or ``author (joined with article)``."""
    names = _table_names(from_clause)
    if len(names) == 1:
        return names[0]
    return f"{names[0]} (joined with {', '.join(names[1:])})"


def _table_names(from_clause: FromClause) -> list[str]:
    if isinstance(from_clause, Join):
        return _table_names(from_clause.left) + _table_names(from_clause.right)
    name = getattr(from_clause, "name", None)
    return [name if _is_given_name(name) else "a subquery"]


def _is_given_name(name: object) -> TypeGuard[str]:
    """Whether a FROM element has a name of its own, not an unnamed subquery's placeholder."""
    return isinstance(name, str) and not isinstance(name, _anonymous_label)
