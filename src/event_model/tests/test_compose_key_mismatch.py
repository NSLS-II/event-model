import pytest

import event_model


def descriptor_bundle():
    return event_model.compose_run().compose_descriptor(
        name="primary",
        data_keys={
            name: {"dtype": "number", "shape": [], "source": "test"}
            for name in ("zulu", "common", "alpha")
        }
        | {
            "streamed": {
                "dtype": "array",
                "shape": [2],
                "source": "test",
                "external": "STREAM:",
            }
        },
    )


@pytest.mark.parametrize("page", [False, True])
@pytest.mark.parametrize(
    "data_fields,timestamp_fields",
    [
        (("common",), ("zulu", "common", "alpha")),
        (("zulu", "common", "alpha"), ("common",)),
        (("common", "alpha"), ("common", "zulu")),
        (("common",), ("common",)),
    ],
)
def test_compose_identifies_mismatched_keys(page, data_fields, timestamp_fields):
    bundle = descriptor_bundle()
    compose = bundle.compose_event_page if page else bundle.compose_event
    value = [0] if page else 0
    data = dict.fromkeys(data_fields, value)
    timestamps = dict.fromkeys((*timestamp_fields, "streamed"), value)

    with pytest.raises(event_model.EventModelValidationError) as exc:
        compose(data=data, timestamps=timestamps)

    assert str(exc.value).splitlines()[:2] == [
        'These sets of keys must match (other than "STREAM:" keys):',
        "Mismatched keys: ['alpha', 'zulu']",
    ]
    # Failed validation must not consume an event sequence number.
    fields = ("zulu", "common", "alpha")
    doc = compose(
        data=dict.fromkeys(fields, value),
        timestamps=dict.fromkeys(fields, value),
    )
    assert doc["seq_num"] == ([1] if page else 1)


@pytest.mark.parametrize("page", [False, True])
@pytest.mark.parametrize("include_streamed", [False, True])
def test_compose_matching_keys_still_excludes_streamed_keys(page, include_streamed):
    bundle = descriptor_bundle()
    compose = bundle.compose_event_page if page else bundle.compose_event
    value = [0] if page else 0
    data = dict.fromkeys(("zulu", "common", "alpha"), value)
    if include_streamed:
        data["streamed"] = value
    timestamps = dict.fromkeys(("alpha", "common", "zulu"), value)

    doc = compose(data=data, timestamps=timestamps)

    assert doc["data"] == data
    assert doc["timestamps"] == timestamps
    assert doc["seq_num"] == ([1] if page else 1)


@pytest.mark.parametrize("page", [False, True])
def test_compose_key_validation_can_still_be_disabled(page):
    bundle = descriptor_bundle()
    compose = bundle.compose_event_page if page else bundle.compose_event
    value = [0] if page else 0

    doc = compose(data={"common": value}, timestamps={"common": value}, validate=False)

    assert doc["data"] == {"common": value}
    assert doc["seq_num"] == ([1] if page else 1)
