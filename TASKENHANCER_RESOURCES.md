# TaskEnhancer Resources — Full Documentation

This appendix provides the complete documentation of the TaskEnhancer
component's internal resources and construction methodology, as requested
by reviewers. All information documented here is directly reproducible
from the code in `model_simple/task_enhancer.py` and
`model_simple/mine_domain_keywords.py`.

---

## 1. Role Lexicon

TaskEnhancer strips agent/role words from task names and descriptions
before semantic matching, so that queries reflect actions rather than
organizational actors.

### 1.1 Complete role word list

The list is defined as `TaskEnhancer.ROLE_WORDS` in `task_enhancer.py`
(lines containing `ROLE_WORDS = [...]`):

```
secretary          the secretary
manager            the manager
department head    logistics department head
logistics head
warehousemen       the warehousemen
warehouse staff
staff              worker          employee
clerk              agent           operator
head               officer         supervisor
coordinator        director        analyst
```

### 1.2 How role words are applied

The method `_remove_role_words(text)` removes each role word from the
text in three ways:

1. ` by <role>` — removes passive-voice attributions;
2. `<role> ` at the start of a phrase — removes leading subjects;
3. ` <role> ` surrounded by spaces — removes mid-sentence occurrences.

### 1.3 Construction rationale

The lexicon was constructed manually to cover the actor vocabulary that
appears in standard BPMN task-naming conventions (swimlane actors,
department heads, warehouse roles). It is intentionally **domain-agnostic**:
it targets grammatical subject positions (who performs the task) rather
than domain terminology, and therefore transfers without modification to
any BPMN scenario where tasks are written as "Actor verbs object."

---

## 2. Passive-to-Active Transformation Rules

### 2.1 Complete rule set

Defined as `TaskEnhancer.PASSIVE_PATTERNS` and expanded in
`_passive_to_active(text)`:

| Pattern (regex) | Replacement | Example |
|---|---|---|
| `\b(is\|are\|was\|were)\s+(\w+ed)\s+by\b` | `\2` | "label is written by secretary" → "written" |
| `\b(is\|are\|was\|were)\s+(\w+en)\s+by\b` | `\2` | "goods are taken by staff" → "taken" |
| `\b(is\|are\|was\|were)\s+(\w+ed)\b` | `\2` | "package is packaged" → "packaged" |
| `\b(is\|are\|was\|were)\s+(\w+en)\b` | `\2` | "form is written" → "written" |

Rules are applied in sequence via `re.sub` with `IGNORECASE`. The
transformation extracts the past-participle verb, which is subsequently
normalized by `_standardize_verbs` (see §3). This two-step design means
the passive→active rules only need to identify the verbal group, not
produce a fully grammatical active sentence.

### 2.2 Construction rationale

The four patterns cover the standard English passive constructions
(copula + past participle, with and without explicit agent "by" phrase),
using regular-expression over *morphological suffixes* (`-ed`, `-en`)
rather than a full POS tagger. This keeps the transformation deterministic
and dependency-free. Edge cases (irregular past participles not ending in
`-ed`/`-en`, e.g., "is sent") are left unchanged rather than
mis-transformed, because the downstream verb standardization table covers
many common irregulars directly (e.g., `'send'` → `'send'`).

---

## 3. Verb Standardization Table

### 3.1 Complete mapping

Defined as `TaskEnhancer.VERB_MAPPING` in `task_enhancer.py`. The table
maps surface-form verbs (including inflected forms) to a canonical BPMN
verb. The full table is reproduced below, grouped by semantic class:

**Decision / Evaluation**

| Input form(s) | Canonical verb |
|---|---|
| clarify, clarifies | clarify |
| determine, determines | determine |
| decide, decides | decide |
| assess, assesses | assess |
| evaluate, evaluates | evaluate |
| check, checks | check |
| verify, verifies | verify |
| validate, validates | validate |
| inspect, inspects | inspect |
| review, reviews | review |

**Computation**

| Input form(s) | Canonical verb |
|---|---|
| calculate, calculates | calculate |
| compute, computes | compute |
| score, scores | score |

**Approval / Rejection**

| Input form(s) | Canonical verb |
|---|---|
| approve, approves, approved | approve |
| reject, rejects, rejected | reject |
| confirm, confirms | confirm |

**Submission / Reception**

| Input form(s) | Canonical verb |
|---|---|
| submit, submits | submit |
| receive, receives | receive |
| register, registers | register |

**Assignment / Scheduling**

| Input form(s) | Canonical verb |
|---|---|
| assign, assigns | assign |
| schedule, schedules | schedule |
| allocate, allocates | allocate |

**Monitoring / Tracking**

| Input form(s) | Canonical verb |
|---|---|
| monitor, monitors | monitor |
| track, tracks | track |
| update, updates | update |

**Cancellation / Termination**

| Input form(s) | Canonical verb |
|---|---|
| cancel, cancels | cancel |
| terminate, terminates | terminate |

**Creation**

| Input form(s) | Canonical verb |
|---|---|
| write, writes, written | write |
| create, creates | create |
| generate, generates | generate |

**Processing**

| Input form(s) | Canonical verb |
|---|---|
| prepare, prepares, prepared | prepare |
| process, processes | process |
| handle, handles | handle |
| perform, performs; do, does | perform |
| execute, executes | execute |

**Communication**

| Input form(s) | Canonical verb |
|---|---|
| invite, invites; request, requests; ask, asks | request |
| notify, notifies | notify |
| inform, informs | inform |
| send, sends | send |

**Selection**

| Input form(s) | Canonical verb |
|---|---|
| select, selects; pick, picks | select |
| choose, chooses | choose |

**Retrieval**

| Input form(s) | Canonical verb |
|---|---|
| take, takes, taken; obtain, obtains | obtain |
| get, gets | get |
| retrieve, retrieves; fetch, fetches | retrieve |

**Packaging / Shipping**

| Input form(s) | Canonical verb |
|---|---|
| package, packages, packaged | package |
| ship, ships | ship |
| deliver, delivers | deliver |

**Other**

| Input form(s) | Canonical verb |
|---|---|
| make, makes | make |
| arrange, arranges | arrange |
| complete, completes | complete |
| finalize, finalizes | finalize |

### 3.2 How the table is applied

`_standardize_verbs(text)` splits the text into tokens and replaces only
the **first** token whose lowercased, punctuation-stripped form appears in
`VERB_MAPPING`. This means standardization targets the task's **leading
verb** (the primary action), leaving object-position verbs unchanged to
avoid corrupting phrases like "check insurance coverage."

### 3.3 Construction rationale

The verb set was derived from the vocabulary of standard BPMN task
descriptions (ISO/IEC 19510 examples and common textbook process models)
and cross-checked against the task names present in the four evaluated
scenarios. Each inflected form is listed explicitly rather than relying
on a stemmer, which ensures the mapping is fully inspectable and produces
no unintended normalizations. The table is designed for transfer: because
it captures *action semantics* (what the task does) rather than
domain-specific terminology, it requires no modification when applied to
new process domains.

---

## 4. Domain Keyword Mapping

### 4.1 Two-tier structure

TaskEnhancer operates with two keyword sources, selected at runtime:

**Tier 1 — Automatically mined keyword table (`domain_keywords_auto.json`)**

Generated offline by `mine_domain_keywords.py` from the ProgrammableWeb
API repository metadata. This is the **default and preferred** source.
Structure of each entry:

```json
"<category_tag>": {
    "injection": "<category_tag> API (word1 word2 word3)",
    "top_words": ["word1", "word2", "word3", ...],
    "api_count": <int>,
    "is_generic": <bool>
}
```

To regenerate:

```bash
python mine_domain_keywords.py \
    --name        api_name.json \
    --description api_description.json \
    --category    api_category.json \
    --output      domain_keywords_auto.json \
    --min-apis    8 \
    --top-k       5
```

**Tier 2 — Hand-written fallback table (`_FALLBACK_DOMAIN_KEYWORDS`)**

Used only when `domain_keywords_auto.json` is absent. Covers ~70 trigger
words across logistics, insurance, warehousing, quoting, ordering,
customer service, complaints, credit/finance, food service, and workflow
domains. The full table is defined in `task_enhancer.py` under
`_FALLBACK_DOMAIN_KEYWORDS` and is reproduced in §4.3 below.

### 4.2 Automatic keyword mining methodology

`mine_domain_keywords.py` performs the following steps (no TF-IDF, LDA,
Word2Vec, or neural embedding is used at any stage):

1. **Group by category**: APIs are grouped by their ProgrammableWeb
   `category` labels (multi-label; one API may belong to several groups).

2. **Filter small categories**: categories with fewer than `--min-apis`
   APIs (default: 8) are excluded.

3. **Count word document-frequency**: for each category, count how many
   distinct API entries contain each word in their `name + description`
   text (after lowercasing and stop-word removal). Only words appearing
   in at least `--min-word-df-ratio` × (number of APIs in category)
   distinct entries (and at least `--min-word-df-floor` entries) are
   candidates.

4. **Provider-level deduplication** (v3 addition): a second document-
   frequency threshold is applied per inferred *provider* (first segment
   of the API slug) to prevent single-company brand names from inflating
   counts.

5. **Select top-k representative words**: the `--top-k` (default: 5)
   words with highest document-frequency are stored as `top_words`.

6. **Mark generic categories** (`is_generic`): categories whose `api_count`
   exceeds the `--generic-percentile` (default: 90th) quantile of the
   api\_count distribution across all categories are flagged as "generic."
   Generic categories are deprioritized in matching but not excluded.

7. **Construct injection string**: `"<category_tag> API (top_word1 … top_wordK)"`.

The stop-word list used in step 3 consists of standard English function
words plus a set of web-API-specific generic terms
(`api`, `data`, `service`, `platform`, `application`, `integration`,
`management`, `system`, `tool`, `solution`, `provider`).

### 4.3 Hand-written fallback table (full listing)

The following table lists all trigger keywords and their injection phrases
in `_FALLBACK_DOMAIN_KEYWORDS`:

| Trigger keyword | Injection phrase |
|---|---|
| shipping | logistics carrier API |
| shipment | shipment tracking API |
| ship | transport delivery service |
| dispatch | logistics coordination service |
| delivery | delivery routing API |
| carrier | carrier selection API |
| logistics | logistics management API |
| freight | freight booking service |
| transport | transportation API |
| courier | courier dispatch API |
| pickup | pickup scheduling API |
| route | route optimization API |
| label | label generation API |
| package | packaging fulfillment service |
| parcel | parcel handling API |
| packing | packaging automation service |
| barcode | barcode generation API |
| qr | QR code generation API |
| insurance | insurance coverage API |
| insure | insurance purchase service |
| coverage | coverage calculation API |
| premium | premium calculation API |
| policy | insurance policy API |
| warehouse | warehouse management API |
| storage | inventory storage API |
| inventory | inventory control API |
| stock | stock management API |
| picking | warehouse picking API |
| goods | goods handling API |
| offer | quote comparison API |
| quote | rate pricing API |
| bid | bidding service API |
| price | pricing calculation API |
| cost | cost estimation API |
| rate | rate lookup API |
| order | order management API |
| purchase | purchase transaction API |
| checkout | checkout processing API |
| cart | shopping cart API |
| payment | payment gateway API |
| invoice | invoice generation API |
| receipt | receipt API |
| customer | customer management API |
| client | client service API |
| user | user management API |
| account | account management API |
| profile | user profile API |
| recourse | complaint handling API |
| claim | claim processing API |
| complaint | complaint resolution API |
| dispute | dispute management API |
| refund | refund processing API |
| compensation | compensation calculation API |
| credit | credit assessment API |
| score | scoring engine API |
| scoring | credit scoring API |
| rating | credit rating API |
| loan | loan processing API |
| mortgage | mortgage evaluation API |
| bank | banking service API |
| finance | financial service API |
| risk | risk assessment API |
| fraud | fraud detection API |
| kyc | KYC verification API |
| aml | AML compliance API |
| income | income verification API |
| asset | asset valuation API |
| application | application processing API |
| approve | approval workflow API |
| reject | rejection notification API |
| restaurant | restaurant management API |
| food | food ordering API |
| meal | meal service API |
| dish | dish selection API |
| menu | menu catalog API |
| beverage | beverage ordering API |
| drink | drink selection API |
| dessert | dessert ordering API |
| tray | tray management service |
| cafeteria | cafeteria management API |
| station | service station API |
| buffet | buffet management API |
| cook | kitchen preparation API |
| prepare | preparation service API |
| serve | service delivery API |
| seat | seating management API |
| table | table reservation API |
| workflow | workflow engine API |
| approval | approval management API |
| review | review process API |
| audit | audit trail API |
| compliance | compliance check API |
| regulation | regulatory API |
| document | document management API |
| form | form processing API |
| report | reporting API |
| notification | notification service API |
| alert | alert management API |
| email | email service API |
| sms | SMS notification API |

---

## 5. Stop-Word Handling

Stop words are used at two stages:

### 5.1 In keyword mining (`mine_domain_keywords.py`)

Before counting word document-frequency, each API's name and description
text is lowercased and filtered against a stop-word set. The set
comprises:

- Standard English function words (articles, prepositions, conjunctions,
  pronouns, auxiliary verbs, etc.);
- Web-API-specific generic terms that appear across nearly all categories
  and therefore carry no discriminative signal:
  `api`, `data`, `service`, `platform`, `application`, `integration`,
  `management`, `system`, `tool`, `solution`, `provider`, and similar.

Words shorter than 3 characters are also excluded.

### 5.2 In cross-category generic-word filtering (`task_enhancer.py`)

At `TaskEnhancer.__init__` time, a second layer of stop-word filtering is
applied to the already-mined `top_words`. A word stem is identified as a
"cross-category generic word" if it appears in the `top_words` lists of
at least `max(20, ceil(0.07 × N_categories))` distinct categories
(where `N_categories` is the number of categories loaded from the keyword
table). These generic stems are removed from the **evidence** set used for
disambiguation (see §6), but are **not** removed from the injection
strings — they still appear in the API term injected into the query, just
not used as matching evidence.

The adaptive threshold (7 % of category count, minimum 20) was set so
that terms appearing across more than ≈ 1-in-14 categories are excluded
as non-discriminative. At the 441-category scale of the ProgrammableWeb
data, this corresponds to approximately 31 categories, which empirically
captures obvious generic web-service terms (`site`, `content`, `create`,
`integrate`, `medium`) while retaining category-specific vocabulary.

---

## 6. Re-ranking Thresholds

The rule-based re-ranking in `match_categories()` uses four configurable
thresholds:

### 6.1 Multi-word category partial-match ratio

```python
_PARTIAL_MATCH_MIN_RATIO = 0.5
```

A multi-word category label (e.g., `"credit-cards"`) is eligible for
partial matching if the fraction of its sub-tokens found in the query
text is ≥ 0.5. This is the standard majority-vote threshold; it is not
tuned per scenario.

### 6.2 Evidence requirement for single-word categories

Controlled by the `require_evidence_for_single_word` constructor
parameter (default: `True` when using auto-mined keywords, behaviour
described in `__init__` docstring):

- `True`: a single-word category label must *also* have at least one
  non-generic representative word from its `top_words` present in the
  query text (fail-closed: an empty evidence set counts as "no evidence").
- `False`: a single-word label match on the token alone is sufficient
  (higher recall, higher risk of homonym collision).

### 6.3 Cross-category generic-word threshold

`max(20, ceil(0.07 × N_categories))` — see §5.2 above.

### 6.4 Sorting priority (descending)

When multiple categories match, they are ranked by the following
sort key (all descending):

1. **Full match** before partial match — a category whose every sub-token
   is found in the text ranks above one where only a majority matched.
2. **Non-generic** before generic — `is_generic = False` ranks above
   `is_generic = True`.
3. **More sub-tokens** — a longer, more specific category label (e.g.,
   `"credit-cards"` with 2 sub-tokens) ranks above a shorter one.
4. **Longer label string** — as a tiebreaker among equal sub-token count.

Up to **3** distinct API injection terms are included in the final query
augmentation, de-duplicated by the first word of the injection phrase
(to avoid near-synonymous terms such as `"credit assessment API"` and
`"credit rating API"` both being injected).

---

## 7. Transferability to Unseen Domains

### 7.1 Automatic update procedure

Because the domain keyword table is derived entirely from the API
repository's own category metadata (§4.2), extending TaskEnhancer to a
new domain requires only one step:

```bash
python mine_domain_keywords.py \
    --name        <new_api_name.json> \
    --description <new_api_description.json> \
    --category    <new_api_category.json> \
    --output      domain_keywords_auto.json
```

No manual rule-writing is needed. The role lexicon (§1) and
passive-to-active rules (§2) are domain-agnostic and require no changes.
The verb standardization table (§3) covers standard BPMN action vocabulary
and would need extension only for highly domain-specific verbs not present
in any standard process modelling corpus.

### 7.2 Components that require attention for new domains

| Component | Domain-specific? | Adaptation needed |
|---|---|---|
| Role lexicon | No — covers grammatical roles | None for standard BPMN |
| Passive-to-active rules | No — morphological rules | None |
| Verb standardization | Mostly no | Add verbs specific to new domain if any |
| Domain keyword table | Yes | Re-run `mine_domain_keywords.py` on new repository |
| Cross-category threshold (7 %) | Self-adaptive | Recomputed automatically from new keyword table size |
| Partial-match ratio (0.5) | No | Fixed at majority-vote principle |

### 7.3 Known limitation

The single-word category disambiguation mechanism (§6.2) was designed for
ProgrammableWeb's category tag vocabulary, where common English words such
as `chef` appear as category labels (referring to configuration management
tools) and collide with their everyday meanings. If a different API
repository uses a more controlled category taxonomy, the
`require_evidence_for_single_word = False` setting may produce higher
recall at acceptable precision. We recommend running the ablation
described in the `__init__` docstring on any new repository before
committing to a default.
