# Adjudication Log: Khmer Sentiment Annotation

Purpose: Record disagreements between annotators and document final decisions to improve consistency and update guidelines.

## Process
1. Identify items where label_A != label_B.
2. Review text and annotator notes.
3. Apply guidelines; discuss edge cases (negation, sarcasm, code-mixing, emojis).
4. Decide final label (POS/NEG/NEU) and rationale.
5. Update `annotation/guidelines.md` if a new rule/example is warranted.

## Log Template
- Item ID: <id>
- Text (de-identified): <text>
- Label A: <POS/NEG/NEU>
- Label B: <POS/NEG/NEU>
- Final Label: <POS/NEG/NEU>
- Rationale: <why chosen>
- Edge Cases: <e.g., sarcasm, negation>
- Action Items: <update guidelines/add example>
- Date: <YYYY-MM-DD>

---

## Decisions
- Item ID: 1003
- Text (de-identified): អរគុណច្រើន សេវាកម្មល្អបំផុត 🙄
- Label A: POS
- Label B: NEG
- Final Label: <POS/NEG/NEU>
- Rationale: <why chosen>
- Edge Cases: <e.g., sarcasm, negation>
- Action Items: <update guidelines/add example>
- Date: <YYYY-MM-DD>

- Item ID: 1004
- Text (de-identified): service ok price fair
- Label A: NEU
- Label B: POS
- Final Label: <POS/NEG/NEU>
- Rationale: <why chosen>
- Edge Cases: <e.g., sarcasm, negation>
- Action Items: <update guidelines/add example>
- Date: <YYYY-MM-DD>

- Item ID: 1007
- Text (de-identified): មិនអាក្រក���ទេ តម្លៃសមរម្យ
- Label A: POS
- Label B: NEU
- Final Label: <POS/NEG/NEU>
- Rationale: <why chosen>
- Edge Cases: <e.g., sarcasm, negation>
- Action Items: <update guidelines/add example>
- Date: <YYYY-MM-DD>

Record adjudicated items here using the template above.

## Action Items Register
Track follow-ups from adjudications. Mark as done when applied to guidelines/tools.
- [x] Add sarcasm example to guidelines (from Item ID: 2025-0001) — done in guidelines v1.0

---

Example Entry
- Item ID: 2025-0001
- Text: "អរគុណច្រើន សេវាកម្មល្អបំផុត 🙄"
- Label A: POS
- Label B: NEG
- Final Label: NEG
- Rationale: Sarcastic emoji contradicts literal praise; broader context implies dissatisfaction.
- Edge Cases: sarcasm
- Action Items: Add sarcasm example to guidelines.
- Date: 2025-01-15
