import re


def sanitize_label(value):
    """
    Convert an arbitrary entity type into a Neo4j-safe label.
    """

    if value is None:
        return "Entity"

    value = str(value).strip()

    value = re.sub(
        r"[^a-zA-Z0-9_]",
        "",
        value
    )

    if not value:
        return "Entity"

    return value


def sanitize_rel_type(value):
    """
    Convert relationship names into Neo4j-safe relationship types.
    """

    if value is None:
        return "RELATED_TO"

    value = str(value).strip().upper()

    value = re.sub(
        r"[^A-Z0-9_]",
        "_",
        value
    )

    value = re.sub(
        r"_+",
        "_",
        value
    )

    return value.strip("_") or "RELATED_TO"


def namespaced_entity_id(document_name, entity_id):
    """
    Create a globally unique graph entity ID.
    """

    return f"{document_name}:{entity_id}"
