// =========================================================
// GRAPH SCHEMA
//
// Reconciled against src/schema/canonical_schema.py. The original
// version of this schema modeled every extracted thing as a generic
// (:Entity) node with a dynamic label taken from domain_extractor's
// old entity_type strings (needing APOC's apoc.merge.node/relationship
// procedures, since plain Cypher can't parameterize a label). That
// generic entity-graph export (domain_knowledge.json) doesn't exist
// in the current pipeline - canonical_schema.py's BondDocument has a
// small, FIXED set of relationship-shaped facts (Issuer -> Issue,
// Issue -> Series/Rating/Trustee/LeadManager), so this schema models
// those directly with fixed labels. No APOC dependency needed anymore.
// =========================================================

CREATE CONSTRAINT issuer_name_unique IF NOT EXISTS
FOR (i:Issuer)
REQUIRE i.name IS UNIQUE;

CREATE CONSTRAINT issue_id_unique IF NOT EXISTS
FOR (iss:Issue)
REQUIRE iss.issue_id IS UNIQUE;

CREATE CONSTRAINT series_id_unique IF NOT EXISTS
FOR (s:Series)
REQUIRE s.series_id IS UNIQUE;

// RatingAgency/DebentureTrustee/LeadManager are reusable entities
// (the same trustee, e.g. "Beacon Trusteeship Limited", legitimately
// serves multiple issuers across multiple documents - see
// test_canonical_extractor.py's PFC and IIFL fixtures) - unique by
// name globally, not namespaced per document, so cross-document
// relationship questions ("which issuers has this trustee served?")
// resolve to one node instead of one per document.
CREATE CONSTRAINT rating_agency_name_unique IF NOT EXISTS
FOR (a:RatingAgency)
REQUIRE a.name IS UNIQUE;

CREATE CONSTRAINT trustee_name_unique IF NOT EXISTS
FOR (t:DebentureTrustee)
REQUIRE t.name IS UNIQUE;

CREATE CONSTRAINT lead_manager_name_unique IF NOT EXISTS
FOR (lm:LeadManager)
REQUIRE lm.name IS UNIQUE;

CREATE INDEX issue_document_name_index IF NOT EXISTS
FOR (iss:Issue)
ON (iss.document_name);

CREATE INDEX series_number_index IF NOT EXISTS
FOR (s:Series)
ON (s.series_number);
