# Annotation Guidelines: Khmer Sentiment Analysis (3-Class)

Version: 1.0  |  Language: Khmer (km)  |  Author: [Your Name]

Goal: Consistently label Khmer social media texts as Positive, Negative, or Neutral. Handle edge cases (negation, sarcasm, emojis, code-mixing) and ensure high inter-annotator agreement (target κ ≥ 0.7).

## 1) Label Schema
- Positive (POS): expresses favorable opinion, satisfaction, praise, happiness, gratitude, excitement.
- Negative (NEG): expresses unfavorable opinion, complaint, anger, sadness, disappointment, dislike.
- Neutral (NEU): factual statements, questions, requests, news headlines, or mixed/unclear sentiment.

Primary rule: Label based on the author’s attitude toward the target (product, service, person, event), not your personal opinion.

## 2) Decision Rules
1. Explicit polarity words dominate (e.g., ល្អ/អាក្រក់; ស្រឡាញ់/មិនចូលចិត្ត). If explicit and unambiguous → label accordingly.
2. Negation flips polarity when it scopes over polarity words.
   - Example: "មិនល្អទេ" (not good) �� Negative; "មិនអាក្រក់ទេ" (not bad) → Positive/Neutral depending on context (often mildly positive).
3. Intensifiers (e.g., ជាងគេ, ណាស់, សាហាវ): strengthen polarity but do not change class.
4. Emojis/emoticons: contribute to sentiment if consistent with text (❤️🙂 → Positive; 😡💢 → Negative; 😐🤔 → Neutral/uncertain). If emoji contradicts text, prefer text unless strong sarcasm cues.
5. Code-mixing: English/Khmer mix is allowed; consider sentiment of the whole text (e.g., "service ok" → mildly Positive/Neutral depending on context; "service so bad" → Negative).
6. Questions: If genuine information-seeking without opinion → Neutral. Rhetorical with sentiment (e.g., "មែនទេ? អាក្រក់បែបនេះ!" ) → Negative.
7. Sarcasm/Irony: If clearly sarcastic, invert literal reading when cues exist (e.g., quotes, rolling eyes 🙄, exaggerated praise with negative context). If uncertain → Neutral and mark as "uncertain" in notes.
8. Mixed sentiment: If both positive and negative appear, choose the dominant sentiment. If balanced/unclear → Neutral.
9. Spam/irrelevant/noise: If non-language content or off-topic, mark Neutral and note "noise".

## 3) Examples (Khmer)
Positive:
- "ខ្ញុំស្រឡាញ់ផលិតផលនេះណាស់! ❤️" → POS
- "សេវាកម្មលឿន និងរីករាយ" → POS
- "មិនអាក្រក់ទេ តម្លៃសមរម្យ" → POS/NEU (lean POS if recommendation tone)

Negative:
- "សេវាកម្មអាក្រក់ណាស់ ខកចិត្ត" → NEG
- "មិនពេញចិត្តទេ ទាក់ទងគ្នាយឺត" → NEG
- "so bad, never again 😡" (code-mix) → NEG

Neutral:
- "ថ្ងៃនេះខ្ញុំទៅផ្សារ" → NEU
- "សួស្តី អ្នកណាដឹងអំពីម៉ូឌែលថ្មីទេ?" → NEU (question)
- "ព័ត៌មានថ្មី៖ ក្រុមហ៊ុនបានប្រកាស…" → NEU (news headline)

Sarcasm (label depends on context):
- "អរគុណច្រើន សេវាកម្មល្អបំផុត🙄" → likely NEG (sarcastic)
- "wow amazing update… app crash ទៅតែទៀត" → NEG

Negation:
- "មិនល្អទេ" → NEG
- "មិនអាក្រក់ទេ" → POS/NEU (context-dependent)

Emojis:
- "ល្អណាស់ 😊" → POS
- "ខកចិត្តណាស់ 😞" → NEG
- "…" or "🤔" without clear stance → NEU

## 4) Target and Context
- Target may be explicit (e.g., service, product) or implicit (post context). If the sentence has clear sentiment but unclear target, still label sentiment.
- If a multi-sentence post contains conflicting sentiments, label the overall dominant sentiment.

## 5) Annotation Procedure
1. Read the entire text.
2. Apply decision rules; assign one label: POS, NEG, or NEU.
3. Add optional notes for edge cases (e.g., "sarcasm?", "code-mix", "negation").
4. If truly ambiguous, choose NEU and add a note.
5. Do not infer beyond the text (no external knowledge unless explicitly present).

## 6) Quality Control
- Double annotation: At least 20% of items labeled independently by two annotators.
- Compute Cohen’s kappa (target κ ≥ 0.7). Review disagreements and update guidelines with examples.
- Adjudication: A third reviewer or joint discussion resolves conflicts; record in `annotation/adjudication_log.md`.
- Regular calibration sessions: Review tricky cases and refresh examples.

## 7) Handling Special Tokens
- PII masked by preprocessing (e.g., <URL>, <EMAIL>, <PHONE>, @anon). These tokens do not affect sentiment unless context implies opinion.
- Repeated characters/letters (e.g., ណាស់ssss, 555): treat as intensifiers. Thai-style "555" means laughter; Khmer users sometimes adopt similar patterns; consider Positive if in praise context.

## 8) Class Imbalance Guidance
- Do not try to balance classes manually during annotation. Annotate ground truth. Class balance is addressed during modeling.

## 9) Tooling
- Recommended tool: Label Studio (or similar). See `annotation/label_config.json` for 3-class setup.
- Install and run:
  - `pip install label-studio`
  - `label-studio start`
- Import data (JSON/CSV); map text field to `text` and use labels: POS, NEG, NEU.
- Export annotations as JSON/CSV for downstream processing.

## 10) Privacy and Ethics Reminders
- Do not copy raw user identifiers into notes.
- If text appears to include sensitive personal information, flag and exclude according to the ethics policy.

## 11) Updating the Guidelines
- Maintain a changelog below with dates and examples.

### Changelog
- v1.0: Initial guidelines with Khmer examples and edge cases.
