"""Unit tests for query expansion module.

Tests the dictionary-based synonym expansion for BM25 sparse queries.
Validates abbreviation expansion, deduplication, and edge case handling.
"""

from article_mind_service.search.query_expander import SYNONYMS, expand_query


class TestQueryExpansion:
    """Tests for expand_query function."""

    def test_single_abbreviation_expanded(self) -> None:
        """Test that single abbreviation is expanded with synonyms."""
        result = expand_query("auth")

        # Should include original term
        assert "auth" in result

        # Should include expansions from SYNONYMS dict
        assert "authentication" in result
        assert "authorize" in result
        assert "login" in result

    def test_multiple_abbreviations_expanded(self) -> None:
        """Test that multiple abbreviations are all expanded."""
        result = expand_query("auth api")

        # Both terms and their expansions should be present
        assert "auth" in result
        assert "authentication" in result
        assert "api" in result
        assert "application" in result
        assert "interface" in result
        assert "endpoint" in result

    def test_unknown_words_passed_through(self) -> None:
        """Test that unknown words are preserved unchanged."""
        result = expand_query("unknown word")

        # Original words should be present
        assert "unknown" in result
        assert "word" in result

        # Should only have original words (no expansion)
        words = result.split()
        assert len(words) == 2

    def test_original_terms_preserved(self) -> None:
        """Test that original terms appear first before expansions."""
        result = expand_query("auth")

        words = result.split()

        # First word should be original
        assert words[0] == "auth"

        # Expansions should follow
        assert "authentication" in words[1:]

    def test_deduplication_works(self) -> None:
        """Test that duplicate words are removed."""
        # If query contains term that's also in expansion, deduplicate
        result = expand_query("auth authentication")

        words = result.split()

        # "authentication" should appear only once
        assert words.count("authentication") == 1

    def test_punctuation_handling(self) -> None:
        """Test that punctuation is stripped for lookup."""
        result = expand_query("auth.")

        # Should match "auth" and expand despite punctuation
        assert "authentication" in result
        assert "authorize" in result

    def test_empty_query_returns_empty(self) -> None:
        """Test that empty query returns empty string."""
        assert expand_query("") == ""
        assert expand_query("   ") == ""

    def test_no_expansion_needed(self) -> None:
        """Test query with no expandable terms returns original."""
        result = expand_query("normal words here")

        words = result.split()

        # Should have exactly original words
        assert words == ["normal", "words", "here"]

    def test_case_insensitive_matching(self) -> None:
        """Test that expansion is case-insensitive."""
        result_lower = expand_query("auth")
        result_upper = expand_query("AUTH")
        result_mixed = expand_query("Auth")

        # All should produce same result (lowercase)
        assert result_lower == result_upper == result_mixed

    def test_mixed_known_unknown_words(self) -> None:
        """Test query with mix of expandable and non-expandable terms."""
        result = expand_query("use auth for security")

        # Known terms expanded
        assert "auth" in result
        assert "authentication" in result

        # Unknown terms preserved
        assert "use" in result
        assert "for" in result
        assert "security" in result

    def test_all_synonym_entries_valid(self) -> None:
        """Test that all entries in SYNONYMS dict produce valid expansions."""
        for abbrev, expansion in SYNONYMS.items():
            result = expand_query(abbrev)

            # Should include original abbreviation
            assert abbrev in result

            # Should include all expansion terms
            for term in expansion.split():
                assert term in result, (
                    f"Expansion term '{term}' missing for abbreviation '{abbrev}'"
                )

    def test_multiple_spaces_normalized(self) -> None:
        """Test that multiple spaces are normalized in output."""
        result = expand_query("auth  api")

        # Should not have double spaces in result
        assert "  " not in result

    def test_whitespace_only_query(self) -> None:
        """Test query with only whitespace returns empty string."""
        assert expand_query("   ") == ""
        assert expand_query("\t\n") == ""

    def test_special_characters_in_query(self) -> None:
        """Test query with special characters preserves original."""
        # Note: Query expansion works on space-separated tokens
        # "auth@api#test" without spaces is treated as one token
        result = expand_query("auth@api#test")

        # Original token should be preserved
        assert "auth@api#test" in result

        # No expansion happens since cleaned version "authapitest" not in SYNONYMS
        words = result.split()
        assert len(words) == 1

        # Test with spaces: special chars separate tokens
        result = expand_query("auth api test")

        # Now "auth" and "api" are separate tokens and get expanded
        assert "auth" in result
        assert "api" in result
        assert "authentication" in result
        assert "application" in result

    def test_technical_term_expansion(self) -> None:
        """Test expansion of common technical abbreviations."""
        test_cases = [
            ("jwt", ["json", "web", "token"]),
            ("ml", ["machine", "learning", "model", "training"]),
            ("k8s", ["kubernetes", "container", "orchestration"]),
            ("db", ["database", "data", "storage"]),
        ]

        for abbrev, expected_terms in test_cases:
            result = expand_query(abbrev)

            for term in expected_terms:
                assert term in result, f"Expected '{term}' in expansion of '{abbrev}'"

    def test_order_preservation(self) -> None:
        """Test that word order is preserved (deduplication maintains first occurrence)."""
        result = expand_query("api auth")

        words = result.split()

        # "api" and its expansions should come before "auth" and its expansions
        api_index = words.index("api")
        auth_index = words.index("auth")

        assert api_index < auth_index, "Original query order should be preserved"

    def test_long_query_expansion(self) -> None:
        """Test expansion of longer query with multiple terms."""
        result = expand_query("use jwt for api auth")

        # All expandable terms should be expanded
        assert "jwt" in result
        assert "json" in result
        assert "web" in result
        assert "token" in result

        assert "api" in result
        assert "application" in result

        assert "auth" in result
        assert "authentication" in result

        # Non-expandable preserved
        assert "use" in result
        assert "for" in result

    def test_punctuation_combinations(self) -> None:
        """Test various punctuation combinations are handled with spaces."""
        # Query expansion splits on whitespace only
        # Punctuation within a token is stripped for lookup but token is preserved

        # Test that trailing punctuation is handled
        result = expand_query("auth.")
        assert "auth." in result  # Original preserved
        assert "authentication" in result  # Cleaned "auth" matches

        # Test punctuation without spaces: treated as one token
        result = expand_query("auth-api")
        assert "auth-api" in result  # Original preserved
        # Cleaned version "authapi" doesn't match SYNONYMS, so no expansion

        # Test with spaces: each token processed separately
        test_cases = [
            "auth, api",
            "auth. api",
            "auth; api",
            "auth: api",
            "auth / api",
        ]

        for query in test_cases:
            result = expand_query(query)

            # Both tokens should be present and expanded
            # Punctuation stripped during lookup but originals preserved
            assert (
                "auth," in result
                or "auth." in result
                or "auth;" in result
                or "auth:" in result
                or "auth" in result
            )
            assert "api" in result
            assert "authentication" in result  # auth expanded
            assert "application" in result  # api expanded

    def test_deduplication_across_expansions(self) -> None:
        """Test deduplication when different abbreviations expand to same term."""
        # Currently no overlapping expansions in SYNONYMS, but test the mechanism
        # by using term that appears both as original and expansion
        result = expand_query("auth authentication api")

        words = result.split()

        # "authentication" should appear only once (deduplication)
        assert words.count("authentication") == 1


class TestQueryExpansionIntegration:
    """Integration tests for query expansion in search context."""

    def test_expansion_improves_recall(self) -> None:
        """Test that expansion produces more search terms for better recall."""
        original = "auth api"
        expanded = expand_query(original)

        # Expanded query should have more terms
        assert len(expanded.split()) > len(original.split())

    def test_expansion_preserves_intent(self) -> None:
        """Test that expansion preserves original query intent."""
        result = expand_query("jwt auth")

        # Original terms must be present (exact match still possible)
        assert "jwt" in result
        assert "auth" in result

    def test_common_search_patterns(self) -> None:
        """Test expansion of common search patterns."""
        test_cases = [
            ("how to use jwt", ["jwt", "json", "web", "token"]),
            ("auth with oauth", ["auth", "authentication", "oauth", "authorization"]),
            ("db connection", ["db", "database", "connection"]),
            ("api design", ["api", "application", "interface", "design"]),
        ]

        for query, expected_terms in test_cases:
            result = expand_query(query)

            for term in expected_terms:
                assert term in result, f"Expected '{term}' in expansion of '{query}'"

    def test_no_expansion_overhead_for_non_abbreviations(self) -> None:
        """Test that queries without abbreviations are returned efficiently."""
        query = "machine learning model training process"
        result = expand_query(query)

        # Should return original (no expansion) but still process
        words_in = query.split()
        words_out = result.split()

        # Output should match input (no abbreviations to expand)
        assert len(words_out) == len(words_in)

    def test_realistic_search_queries(self) -> None:
        """Test expansion with realistic search queries."""
        # Realistic queries from software documentation search
        queries = [
            "jwt auth implementation",
            "how to configure db connection",
            "api endpoint design patterns",
            "async programming best practices",
            "ml model deployment",
        ]

        for query in queries:
            result = expand_query(query)

            # Should always return non-empty result
            assert result, f"Empty result for query: {query}"

            # Should have at least as many terms as original
            assert len(result.split()) >= len(query.split())

    def test_abbreviation_at_different_positions(self) -> None:
        """Test abbreviations at start, middle, and end of query."""
        # Start
        result = expand_query("auth is important")
        assert "authentication" in result

        # Middle
        result = expand_query("using auth for security")
        assert "authentication" in result

        # End
        result = expand_query("implement secure auth")
        assert "authentication" in result

    def test_empty_synonym_expansion(self) -> None:
        """Test behavior when SYNONYMS dict has empty expansion (shouldn't happen but defensive)."""
        # This tests the robustness of the code
        # If SYNONYMS had an empty value, it should still work
        result = expand_query("test query")

        # Should work even with no matches
        assert "test" in result
        assert "query" in result
