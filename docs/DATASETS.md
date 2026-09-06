# Datasets

Synthetic dense data is deterministic CC0 project-generated token data and is the controlled
systems workload. WikiText-103 (Creative Commons Attribution-ShareAlike 3.0) is optional pipeline
validation. C4 is limited to a fixed bounded shard/subset and retains ODC-BY/Common Crawl terms.
Criteo Click Logs retain CC BY-NC-SA terms and are skipped if bounded access would pull the full
corpus. Synthetic sparse is explicitly not reported as Criteo.

Every acquired artifact records source, revision, terms, expected/actual bytes, SHA-256, rows,
tokens, transform version, and seed. Raw third-party data is ignored and not redistributed.
