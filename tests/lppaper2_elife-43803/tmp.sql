CREATE TABLE Manuscript (
    id TEXT PRIMARY KEY,
    content TEXT,
    created_at TIMESTAMP
);

CREATE TABLE Prompt (
    id TEXT PRIMARY KEY,
    prompt_text TEXT,
    model TEXT,
    prompt_type TEXT,  -- 'claim_extraction', 'llm_results', 'peer_results', 'comparison'
    created_at TIMESTAMP
);

CREATE TABLE Claim (
    id TEXT PRIMARY KEY,  -- Can use claim_id from JSON
    manuscript_id TEXT,
    claim_id TEXT,  -- e.g., "C1", "C2"
    claim TEXT,
    claim_type TEXT,  -- EXPLICIT or IMPLICIT
    source_text TEXT,
    evidence_type TEXT,  -- JSON array: ["DATA", "CITATION", ...]
    evidence_reasoning TEXT,
    prompt_id TEXT,
    FOREIGN KEY (manuscript_id) REFERENCES Manuscript(id),
    FOREIGN KEY (prompt_id) REFERENCES Prompt(id)
);

CREATE TABLE Peers (
    id TEXT PRIMARY KEY,
    manuscript_id TEXT,
    content TEXT,
    created_at TIMESTAMP,
    FOREIGN KEY (manuscript_id) REFERENCES Manuscript(id)
);

CREATE TABLE Results_LLM (
    id TEXT PRIMARY KEY,  -- Can use result_id from JSON
    manuscript_id TEXT,
    result_id TEXT,  -- e.g., "R1", "R2"
    reviewer_id TEXT,  -- ORCID or "LLM"
    reviewer_name TEXT,
    status TEXT,  -- SUPPORTED, UNSUPPORTED, UNCERTAIN
    status_reasoning TEXT,
    prompt_id TEXT,
    FOREIGN KEY (manuscript_id) REFERENCES Manuscript(id),
    FOREIGN KEY (prompt_id) REFERENCES Prompt(id)
);

CREATE TABLE Results_Peers (
    id TEXT PRIMARY KEY,  -- Can use result_id from JSON
    peer_id TEXT,
    result_id TEXT,  -- e.g., "R1", "R2"
    reviewer_id TEXT,  -- ORCID
    reviewer_name TEXT,
    status TEXT,  -- SUPPORTED, UNSUPPORTED, UNCERTAIN
    status_reasoning TEXT,
    prompt_id TEXT,
    FOREIGN KEY (peer_id) REFERENCES Peers(id),
    FOREIGN KEY (prompt_id) REFERENCES Prompt(id)
);

CREATE TABLE Comparison (
    id TEXT PRIMARY KEY,
    llm_result_id TEXT,  -- nullable
    peer_result_id TEXT,  -- nullable
    llm_status TEXT,  -- nullable
    peer_status TEXT,  -- nullable
    agreement_status TEXT,  -- 'agree', 'disagree', 'partial'
    notes TEXT,  -- nullable
    n_llm INTEGER,  -- nullable
    n_peer INTEGER,  -- nullable
    n_itx INTEGER,  -- nullable (intersection count)
    prompt_id TEXT,
    FOREIGN KEY (llm_result_id) REFERENCES Results_LLM(id),
    FOREIGN KEY (peer_result_id) REFERENCES Results_Peers(id),
    FOREIGN KEY (prompt_id) REFERENCES Prompt(id)
);

-- Junction Tables
CREATE TABLE Claims_ResultsLLM (
    claim_id TEXT,
    result_llm_id TEXT,
    PRIMARY KEY (claim_id, result_llm_id),
    FOREIGN KEY (claim_id) REFERENCES Claim(id),
    FOREIGN KEY (result_llm_id) REFERENCES Results_LLM(id)
);

CREATE TABLE Claims_ResultsPeers (
    claim_id TEXT,
    result_peer_id TEXT,
    PRIMARY KEY (claim_id, result_peer_id),
    FOREIGN KEY (claim_id) REFERENCES Claim(id),
    FOREIGN KEY (result_peer_id) REFERENCES Results_Peers(id)
);