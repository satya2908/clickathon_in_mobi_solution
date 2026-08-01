"""Tests for placeholder substitution and scalar re-typing.

Configuration is the layer where a wrong value does the most damage per line of code: it is
resolved once at startup, far from where it is used, and a mistyped setting surfaces as an
error message that points at the query rather than at the config. The `max_threads` case
below is not hypothetical -- it shipped, and it failed against ClickHouse Cloud with
"Cannot parse input: expected 'eof' before: 'False'".
"""

import pytest

from verdict.config import ConfigError, _coerce, _substitute, expand, read_dotenv


class TestSubstitution:
    def test_resolves_a_plain_placeholder(self):
        assert _substitute("host: ${H}", {"H": "example.com"}) == "host: example.com"

    def test_raises_when_a_required_placeholder_is_unset(self):
        with pytest.raises(ConfigError, match="H"):
            _substitute("host: ${H}", {})

    def test_colon_dash_falls_back_when_unset_or_empty(self):
        assert _substitute("${H:-fallback}", {}) == "fallback"
        assert _substitute("${H:-fallback}", {"H": ""}) == "fallback"

    def test_bare_dash_falls_back_only_when_unset(self):
        """POSIX semantics: an explicitly empty variable is a deliberate choice.

        An operator who sets PASSWORD= to mean "no password" should get no password, not a
        default silently substituted underneath them.
        """
        assert _substitute("${H-fallback}", {}) == "fallback"
        assert _substitute("${H-fallback}", {"H": ""}) == ""

    def test_reports_every_missing_variable_at_once(self):
        """One run should reveal the whole list, not one name per attempt."""
        with pytest.raises(ConfigError) as err:
            _substitute("${A} ${B} ${C}", {"B": "set"})
        message = str(err.value)
        assert "A" in message and "C" in message and "B" not in message.split(":")[-1]


class TestCoercion:
    @pytest.mark.parametrize("text", ["true", "TRUE", "yes", "on", " True "])
    def test_recognises_textual_true(self, text):
        assert _coerce(text) is True

    @pytest.mark.parametrize("text", ["false", "FALSE", "no", "off", " False "])
    def test_recognises_textual_false(self, text):
        assert _coerce(text) is False

    @pytest.mark.parametrize("text", ["0", "1"])
    def test_zero_and_one_are_integers_not_booleans(self, text):
        """The bug that broke `schema apply` against Cloud.

        `max_threads: ${CLICKHOUSE_MAX_THREADS:-0}` resolved to False, which the client sent
        to the server as a setting value. ClickHouse rejected it with a parse error naming
        'False' -- an error whose text gives no hint that the cause is in a YAML default.
        Nothing else in the pipeline would have caught it: False is a valid Python object, it
        passes Pydantic's `dict[str, Any]`, and it only fails at the wire.

        The general rule this encodes: in configuration, 0 and 1 are numbers. A flag that
        wants to be a boolean can spell it.
        """
        value = _coerce(text)
        assert isinstance(value, int) and not isinstance(value, bool)
        assert value == int(text)

    def test_parses_integers_and_floats(self):
        assert _coerce("8443") == 8443
        assert _coerce("0.5") == 0.5
        assert _coerce("-3") == -3

    def test_parses_explicit_nulls(self):
        assert _coerce("null") is None
        assert _coerce("~") is None

    def test_leaves_ordinary_strings_alone(self):
        assert _coerce("ap-south-1.aws.clickhouse.cloud") == "ap-south-1.aws.clickhouse.cloud"

    def test_a_version_string_is_not_mangled_into_a_float(self):
        assert _coerce("1.2.3") == "1.2.3"


class TestExpand:
    def test_walks_nested_structures(self):
        tree = {"a": ["${X}", {"b": "${Y:-2}"}], "c": "${Z:-off}"}
        assert expand(tree, {"X": "1"}) == {"a": [1, {"b": 2}], "c": False}

    def test_leaves_non_string_leaves_untouched(self):
        assert expand({"n": 5, "f": 1.5, "b": True, "z": None}, {}) == {
            "n": 5,
            "f": 1.5,
            "b": True,
            "z": None,
        }

    def test_a_settings_block_survives_round_trip_with_correct_types(self):
        """The exact shape that failed, asserted end to end."""
        settings = expand(
            {"max_execution_time": "${T:-300}", "max_threads": "${N:-0}"},
            {},
        )
        assert settings == {"max_execution_time": 300, "max_threads": 0}
        assert not isinstance(settings["max_threads"], bool)


class TestDotenvIsReadButNeverWins:
    """`.env` was a Docker-only artefact: Compose reads it, the CLI did not.

    Editing a value there and running `verdict investigate` changed nothing, with no error to
    explain the silence. Reading it closes that gap; letting the real environment win keeps a
    Kubernetes secret or an explicit `LLM_ENABLED=false` from being shadowed by a stray file.
    """

    def test_values_are_read_from_the_file(self, tmp_path):
        path = tmp_path / ".env"
        path.write_text("LLM_MODEL=gemini-flash-latest\nLLM_MAX_TOKENS=3000\n")
        assert read_dotenv(path) == {"LLM_MODEL": "gemini-flash-latest", "LLM_MAX_TOKENS": "3000"}

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        assert read_dotenv(tmp_path / "nope.env") == {}

    def test_comments_blanks_and_exports_are_handled(self, tmp_path):
        path = tmp_path / ".env"
        path.write_text("# a comment\n\nexport LLM_MODEL=gpt-4o-mini\n   \nNOT_A_PAIR\n")
        assert read_dotenv(path) == {"LLM_MODEL": "gpt-4o-mini"}

    def test_surrounding_quotes_are_stripped(self, tmp_path):
        path = tmp_path / ".env"
        path.write_text("A=\"quoted\"\nB='single'\nC=bare\n")
        assert read_dotenv(path) == {"A": "quoted", "B": "single", "C": "bare"}

    def test_a_hash_inside_a_value_survives(self, tmp_path):
        """Truncating a credential at a '#' is a miserable failure to diagnose."""
        path = tmp_path / ".env"
        path.write_text("LLM_API_KEY=abc#def\n")
        assert read_dotenv(path)["LLM_API_KEY"] == "abc#def"

    def test_an_equals_sign_inside_a_value_survives(self, tmp_path):
        path = tmp_path / ".env"
        path.write_text("TOKEN=a=b=c\n")
        assert read_dotenv(path)["TOKEN"] == "a=b=c"
