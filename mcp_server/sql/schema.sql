-- =========================================================
-- inrbonds MySQL schema
--
-- Reconciled against src/schema/canonical_schema.py (BondDocument).
-- The original version of this schema (written before
-- canonical_schema.py existed) only covered `series` /
-- `series_category_terms` - i.e. BondSeries/InvestorCategory. It had
-- no tables at all for Rating, debenture_trustee, lead_managers, or
-- any of IssueTerms (issue_size, shelf_limit, green_shoe_option,
-- security_cover, security_type, ...) - those are all real fields on
-- BondIssue/IssueTerms that `hoist_shared_series_terms()` moves OUT
-- of series-level data, so a loader reading only the old
-- `series_terms.json` never saw them at all.
--
-- IssueTerms.additional_terms (a free-form Dict[str, Fact]) is
-- intentionally NOT modeled here - it's an open-ended bag of
-- less-common facts that don't fit a fixed relational column set.
-- It stays queryable via Qdrant/Neo4j instead.
-- =========================================================

CREATE DATABASE IF NOT EXISTS inrbonds;

USE inrbonds;

-- =========================================================
-- 1. ISSUERS
-- =========================================================

CREATE TABLE IF NOT EXISTS issuers (
    issuer_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    issuer_name VARCHAR(255) NOT NULL,
    issuer_short_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uq_issuer_name
        UNIQUE (issuer_name)
);


-- =========================================================
-- 2. DOCUMENTS
-- =========================================================

CREATE TABLE IF NOT EXISTS documents (
    document_id BIGINT AUTO_INCREMENT PRIMARY KEY,

    issuer_id BIGINT NOT NULL,

    document_name VARCHAR(500) NOT NULL,
    document_type VARCHAR(100) NOT NULL,

    -- document.document_id from canonical_schema.py's DocumentMetadata
    -- (e.g. "pfc_ncd_2026") - distinct from this table's own
    -- auto-increment PK, kept so loaders can re-run idempotently
    -- against the same canonical JSON without depending on row order.
    document_identifier VARCHAR(255),
    document_date VARCHAR(100),
    page_count INT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_document_issuer
        FOREIGN KEY (issuer_id)
        REFERENCES issuers(issuer_id)
        ON DELETE CASCADE,

    CONSTRAINT uq_document
        UNIQUE (issuer_id, document_name)
);


CREATE INDEX idx_documents_issuer
    ON documents(issuer_id);

CREATE INDEX idx_documents_type
    ON documents(document_type);


-- =========================================================
-- 3. ISSUES / OFFERS
--
-- Columns below mirror IssueTerms' fixed fields directly (a wide
-- table, not EAV) since IssueTerms is a bounded, named set of Facts -
-- one row per BondIssue is the natural shape. debenture_trustee is
-- here too (not a separate table) since canonical_schema.py models it
-- as a single Fact per issue, never a list (SEBI NCS Regulations
-- mandate exactly one trustee per issue).
-- =========================================================

CREATE TABLE IF NOT EXISTS issues (
    issue_id BIGINT AUTO_INCREMENT PRIMARY KEY,

    document_id BIGINT NOT NULL,

    -- BondIssue.issue_id from canonical_schema.py (e.g.
    -- "pfc_ncd_2026_issue_1") - lets the loader upsert idempotently.
    issue_identifier VARCHAR(255),
    issue_name VARCHAR(500),
    issue_type VARCHAR(100),

    -- --- IssueTerms fixed fields ---
    issue_size VARCHAR(255),
    issue_open_date VARCHAR(100),
    issue_close_date VARCHAR(100),
    allotment_date VARCHAR(100),
    listing VARCHAR(500),
    exchange VARCHAR(255),
    security_type VARCHAR(100),
    shelf_limit VARCHAR(255),
    green_shoe_option VARCHAR(255),
    security_cover VARCHAR(50),

    -- --- BondIssue.debenture_trustee (single Fact, not a list) ---
    debenture_trustee VARCHAR(500),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_issue_document
        FOREIGN KEY (document_id)
        REFERENCES documents(document_id)
        ON DELETE CASCADE,

    CONSTRAINT uq_issue_identifier
        UNIQUE (document_id, issue_identifier)
);


CREATE INDEX idx_issues_document
    ON issues(document_id);


-- =========================================================
-- 4. RATINGS
--
-- BondIssue.ratings: List[Rating]. Did not exist in the original
-- schema at all - ratings were only ever reachable via Qdrant's prose
-- chunks (lossy for exact lookups like "what is PFC's CRISIL rating").
-- =========================================================

CREATE TABLE IF NOT EXISTS ratings (
    rating_id BIGINT AUTO_INCREMENT PRIMARY KEY,

    issue_id BIGINT NOT NULL,

    agency VARCHAR(255) NOT NULL,
    rating VARCHAR(500) NOT NULL,
    outlook VARCHAR(100),
    instrument VARCHAR(100),
    rating_date VARCHAR(100),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_rating_issue
        FOREIGN KEY (issue_id)
        REFERENCES issues(issue_id)
        ON DELETE CASCADE,

    CONSTRAINT uq_issue_agency_rating
        UNIQUE (issue_id, agency, rating)
);

CREATE INDEX idx_ratings_issue
    ON ratings(issue_id);

CREATE INDEX idx_ratings_agency
    ON ratings(agency);


-- =========================================================
-- 5. LEAD MANAGERS
--
-- BondIssue.lead_managers: List[Fact] - unlike debenture_trustee,
-- several are routinely appointed together, so this is a proper
-- one-to-many table, not a column on `issues`.
-- =========================================================

CREATE TABLE IF NOT EXISTS lead_managers (
    lead_manager_id BIGINT AUTO_INCREMENT PRIMARY KEY,

    issue_id BIGINT NOT NULL,
    name VARCHAR(500) NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_lead_manager_issue
        FOREIGN KEY (issue_id)
        REFERENCES issues(issue_id)
        ON DELETE CASCADE,

    CONSTRAINT uq_issue_lead_manager
        UNIQUE (issue_id, name)
);

CREATE INDEX idx_lead_managers_issue
    ON lead_managers(issue_id);


-- =========================================================
-- 6. SERIES
-- =========================================================

CREATE TABLE IF NOT EXISTS series (
    series_id BIGINT AUTO_INCREMENT PRIMARY KEY,

    issue_id BIGINT NOT NULL,

    -- BondSeries.series_id from canonical_schema.py (e.g.
    -- "pfc_ncd_2026_series_III") - distinct from series_code
    -- (BondSeries.series_number, e.g. "III").
    series_identifier VARCHAR(255),
    series_code VARCHAR(50) NOT NULL,

    tenor VARCHAR(100),
    frequency VARCHAR(100),
    -- Normalized classification derived from `frequency`
    -- (SeriesTerms.coupon_type, e.g. "Zero Coupon", "Cumulative"),
    -- kept alongside the raw `frequency` value rather than replacing it.
    coupon_type VARCHAR(100),

    face_value_inr DECIMAL(18,2),

    mode_of_interest_payment VARCHAR(500),

    maturity_redemption VARCHAR(500),

    nature_of_indebtedness VARCHAR(255),

    put_call_option VARCHAR(500),

    minimum_application TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_series_issue
        FOREIGN KEY (issue_id)
        REFERENCES issues(issue_id)
        ON DELETE CASCADE,

    CONSTRAINT uq_issue_series
        UNIQUE (issue_id, series_code)
);


CREATE INDEX idx_series_issue
    ON series(issue_id);

CREATE INDEX idx_series_code
    ON series(series_code);

CREATE INDEX idx_series_tenor
    ON series(tenor);


-- =========================================================
-- 7. INVESTOR CATEGORIES
--
-- Always atomic codes (I/II/III/IV/...), never a compound code like
-- "I_II_III_IV" - table_extractor.py's extract_category() can return
-- an arbitrary-length compound code when one source row covers
-- several categories at once (see its CATEGORY_NUMERAL_ORDER /
-- Capri Global regression test); the loader splits that into one row
-- per atomic category before writing here (see
-- load_canonical_to_mysql.py's split_category_id()).
-- =========================================================

CREATE TABLE IF NOT EXISTS investor_categories (
    category_id BIGINT AUTO_INCREMENT PRIMARY KEY,

    category_code VARCHAR(50) NOT NULL,
    category_name VARCHAR(255) NOT NULL,

    CONSTRAINT uq_category_code
        UNIQUE (category_code)
);


-- =========================================================
-- 8. SERIES x INVESTOR CATEGORY TERMS
-- =========================================================

CREATE TABLE IF NOT EXISTS series_category_terms (
    series_category_term_id BIGINT AUTO_INCREMENT PRIMARY KEY,

    series_id BIGINT NOT NULL,
    category_id BIGINT NOT NULL,

    issue_price_inr DECIMAL(18,2),

    coupon_percent DECIMAL(10,4),

    effective_yield_percent DECIMAL(10,4),

    maturity_amount_inr DECIMAL(18,2),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_terms_series
        FOREIGN KEY (series_id)
        REFERENCES series(series_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_terms_category
        FOREIGN KEY (category_id)
        REFERENCES investor_categories(category_id)
        ON DELETE RESTRICT,

    CONSTRAINT uq_series_category
        UNIQUE (series_id, category_id)
);


CREATE INDEX idx_terms_series
    ON series_category_terms(series_id);

CREATE INDEX idx_terms_category
    ON series_category_terms(category_id);

CREATE INDEX idx_terms_coupon
    ON series_category_terms(coupon_percent);


-- =========================================================
-- 9. PROVENANCE
--
-- One row per (series, page, source_field) - source_field distinguishes
-- which Fact on that series/category the page supports (e.g. "tenor",
-- "coupon:III") since a series can have several Facts sourced from
-- different pages.
-- =========================================================

CREATE TABLE IF NOT EXISTS series_provenance (
    provenance_id BIGINT AUTO_INCREMENT PRIMARY KEY,

    series_id BIGINT NOT NULL,

    page_number INT NOT NULL,

    source_field VARCHAR(100),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_provenance_series
        FOREIGN KEY (series_id)
        REFERENCES series(series_id)
        ON DELETE CASCADE,

    CONSTRAINT uq_series_page_field
        UNIQUE (series_id, page_number, source_field)
);


CREATE INDEX idx_provenance_series
    ON series_provenance(series_id);

CREATE INDEX idx_provenance_page
    ON series_provenance(page_number);


-- =========================================================
-- DEFAULT INVESTOR CATEGORIES
--
-- Just a convenience seed for the common case - get_or_create_category()
-- in load_canonical_to_mysql.py creates any other atomic code
-- (e.g. a document with only "I"/"II" or a document with a "V") on
-- demand, so this list is not a hard constraint.
-- =========================================================

INSERT IGNORE INTO investor_categories
    (category_code, category_name)
VALUES
    ('I', 'Category I'),
    ('II', 'Category II'),
    ('III', 'Category III'),
    ('IV', 'Category IV');
