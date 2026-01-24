"""Unit tests for metadata filtering in dense search."""

import pytest

from article_mind_service.search.dense_search import DenseSearch


class TestMetadataFiltering:
    """Tests for metadata filter building."""

    def test_build_where_clause_single_filter(self):
        """Test building where clause with single filter."""
        dense = DenseSearch()

        where = dense._build_where_clause({"article_id": 42})

        assert where == {"article_id": {"$eq": 42}}

    def test_build_where_clause_multiple_filters(self):
        """Test building where clause with multiple filters."""
        dense = DenseSearch()

        where = dense._build_where_clause({"article_id": 42, "has_code": True})

        # Should use $and operator
        assert "$and" in where
        assert len(where["$and"]) == 2
        assert {"article_id": {"$eq": 42}} in where["$and"]
        assert {"has_code": {"$eq": True}} in where["$and"]

    def test_build_where_clause_empty_filters(self):
        """Test building where clause with empty filters."""
        dense = DenseSearch()

        where = dense._build_where_clause({})

        assert where == {}

    def test_build_where_clause_none_filters(self):
        """Test building where clause with None filters."""
        dense = DenseSearch()

        where = dense._build_where_clause(None)

        assert where == {}

    def test_build_where_clause_string_value(self):
        """Test building where clause with string value."""
        dense = DenseSearch()

        where = dense._build_where_clause({"source_url": "https://example.com"})

        assert where == {"source_url": {"$eq": "https://example.com"}}

    def test_build_where_clause_numeric_value(self):
        """Test building where clause with numeric value."""
        dense = DenseSearch()

        where = dense._build_where_clause({"word_count": 100})

        assert where == {"word_count": {"$eq": 100}}

    def test_build_where_clause_boolean_value(self):
        """Test building where clause with boolean value."""
        dense = DenseSearch()

        where = dense._build_where_clause({"has_code": True})

        assert where == {"has_code": {"$eq": True}}

    def test_build_where_clause_mixed_types(self):
        """Test building where clause with mixed value types."""
        dense = DenseSearch()

        where = dense._build_where_clause({
            "article_id": 42,
            "has_code": False,
            "source_url": "https://example.com"
        })

        # Should use $and operator with all filters
        assert "$and" in where
        assert len(where["$and"]) == 3
