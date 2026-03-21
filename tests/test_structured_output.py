"""Tests for structured output JSON schema generation."""

from src.llm.models import (
    LightScanResponse,
    DeepTriggerResponse,
    light_scan_schema,
    deep_trigger_schema,
)


class TestSchemaGeneration:
    """Tests for JSON Schema generators."""

    def test_light_scan_schema_has_required_fields(self):
        schema = light_scan_schema()
        assert "properties" in schema
        props = schema["properties"]
        assert "sentiment_delta" in props
        assert "uncertainty_penalty" in props
        assert "rationale" in props

    def test_deep_trigger_schema_has_required_fields(self):
        schema = deep_trigger_schema()
        assert "properties" in schema
        props = schema["properties"]
        assert "sentiment_delta" in props
        assert "uncertainty_penalty" in props
        assert "key_factors" in props
        assert "rationale" in props

    def test_light_scan_schema_is_valid_json_schema(self):
        schema = light_scan_schema()
        assert schema.get("type") == "object"
        assert "title" in schema

    def test_deep_trigger_schema_is_valid_json_schema(self):
        schema = deep_trigger_schema()
        assert schema.get("type") == "object"

    def test_schema_roundtrip_light(self):
        """Validate that a conforming dict passes Pydantic validation."""
        data = {
            "sentiment_delta": 0.15,
            "uncertainty_penalty": 0.3,
            "rationale": "Test rationale",
        }
        obj = LightScanResponse(**data)
        assert obj.sentiment_delta == 0.15

    def test_schema_roundtrip_deep(self):
        data = {
            "sentiment_delta": -0.2,
            "uncertainty_penalty": 0.5,
            "key_factors": ["factor1", "factor2"],
            "rationale": "Test deep rationale",
        }
        obj = DeepTriggerResponse(**data)
        assert obj.key_factors == ["factor1", "factor2"]
