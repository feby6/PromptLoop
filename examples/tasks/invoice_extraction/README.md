# invoice_extraction

Messy invoice and receipt text in several languages and currencies → JSON with
`vendor`, `invoice_number`, `date`, `total`, `currency`.

The task description deliberately leaves out the conventions the expected outputs
follow, so a good prompt has to infer them from the examples:

- `date` in ISO format (`YYYY-MM-DD`), whatever the source format
- `total` is the final amount payable (after tax/discounts) as a plain number
- `currency` as an ISO 4217 code (`$` → `USD`, `A$` → `AUD`, `₹`/`Rs.` → `INR`, ...)
- `invoice_number` is `null` when the document has none

Scored with `json_match` (per-field accuracy). All examples are hand-written.
