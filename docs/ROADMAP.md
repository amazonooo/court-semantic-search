# Court Semantic Search — MVP roadmap

## Product principle

The system searches by factual/legal similarity first and only then explains the outcome of each case. A user's preferred result (for example, only successful or only unsuccessful cases) must not silently become a retrieval bias. Outcome is metadata/analysis of a relevant case, not a default hard filter.

## 1. Retrieval: hard filters first

Before semantic search/reranking, apply structured constraints that can be checked deterministically:

- period (`dateFrom`, `dateTo`);
- court;
- participant/company (Parser API `inn`, which also accepts participant name/FIO according to provider docs);
- dispute type/category;
- known case number when supplied;
- canonical case identity after discovery: `CaseId`.

### Date rules

A user-provided period is a hard boundary for entering the candidate set.

Examples for the future query-analyzer layer:

- `после 2020` -> `dateFrom=2021-01-01`;
- `с 2020 по 2023` -> `dateFrom=2020-01-01`, `dateTo=2023-12-31`.

The backend must re-check document dates after the provider response instead of trusting only the upstream filter. If a date restriction exists and a candidate document has no verifiable registration date, it must not enter the candidate set.

Once a case has legitimately matched inside the requested period, case expansion may attach its full procedural history (including acts outside the period). Those extra acts provide context and do not cause the case to enter the result set.

## 2. CaseId as the canonical case identity

`CaseId` identifies the whole case, while `FileUrl`/`document_id` identifies a judicial act.

Flow:

1. initial RAS search finds candidate acts;
2. deduplicate acts by `document_id`;
3. group candidates by `CaseId`;
4. because Parser API does not expose a direct `caseId` search parameter, expand each matched case by its base `caseNumber`;
5. keep only returned documents with the exact original `CaseId`;
6. attach all instances/acts to one `CourtCase`.

## 3. Document roles inside a case

Do not treat one universal `preferred_document` as the final architecture. Replace it with explicit roles:

- first-instance decision / main facts document;
- appellate acts;
- cassation acts;
- latest substantive act;
- procedural acts;
- later factual/evidentiary additions.

The first instance is the primary factual source because it usually contains the fullest description of circumstances and evidence. Higher instances remain searchable and must be attached because new evidence/facts can appear later and because they determine whether the result was changed, cancelled or upheld.

## 4. Search from general to specific

The intended MVP pipeline is multi-stage:

1. hard filters;
2. broad legal/factual signals;
3. structured feature narrowing (roughly 10–20 MVP features), including:
   - relationship/type of dispute;
   - claim subject;
   - events such as assignment, reorganization, succession;
   - period;
   - court;
   - instance;
   - applied norms;
   - outcome as a descriptive feature, not a default retrieval filter;
4. semantic search/reranking;
5. final verification of a short 5–10 case shortlist against the user's real criteria;
6. analysis of whether legislation or judicial approach changed over time when relevant.

## 5. Reusable collections / monitoring scenarios

Support reusable court-case collections built from structured filters, for example:

- disputes of a particular company/participant;
- INN-based collection;
- court-specific collection;
- region-based collection (requires a region -> courts resolver);
- period-based collection combined with another supported provider criterion;
- dispute type/category;
- combinations of the above.

Current RAS provider does not expose a dedicated OGRN parameter, so OGRN must not be silently mapped to another field. Add it only after provider/KAD capability is verified.

"Current proceedings" monitoring is a separate scenario: RAS is act-oriented; reliable current status, participants, hearings and procedural history should be enriched from KAD when that provider is added.

## 6. Next implementation order

1. Date hard-filter re-check after RAS response. [in progress]
2. High-level collection search endpoint for participant/court/period/category combinations. [in progress]
3. Replace `preferred_document` with explicit document roles. [done]
4. Multi-query retrieval: accept several search formulations, union, deduplicate and group by `CaseId`.
5. First-instance-first factual retrieval while still allowing higher-instance matches. [done]
6. KAD provider for case status, participants and full procedural history.
7. Region -> courts resolver and region collections.
8. OGRN support only if confirmed by provider/KAD.
9. LLM/semantic stages owned by the corresponding team member: query decomposition, reranking, final checklist, summaries and change-of-law/practice analysis.
