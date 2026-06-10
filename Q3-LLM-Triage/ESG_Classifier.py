# ============================================================
# BUS5001 Assessment 3 - Q3: ESG Message Triage
# Student ID: 22873257
# Description: ESG message classification using Hugging Face
# zero-shot classifier (baseline) and LLM (Claude) comparison
# ============================================================

from transformers import pipeline
import json

# ============================================================
# IMPROVED PROMPT TEMPLATE (Q3a)
# ============================================================
IMPROVED_PROMPT = """
You are an ESG operations triage analyst. Analyse the message 
and return ONLY valid JSON with these fields:
- issue_category: [Environmental, Social, Governance, 
  Facilities, Procurement, Accessibility]
- urgency: LOW, MEDIUM, HIGH, or CRITICAL
- sentiment: POSITIVE, NEUTRAL, or NEGATIVE
- followup_required: Y or N
- recommended_team: most appropriate team
- escalation_reason: brief reason or No escalation required
- data_sensitivity_risk: LOW, MEDIUM, or HIGH
- brief_summary: one sentence summary

Urgency guide:
- CRITICAL: immediate safety or legal risk
- HIGH: significant ongoing damage or barrier
- MEDIUM: recurring issue, moderate impact
- LOW: minor, one-off issue
"""

# ============================================================
# ESG TEST MESSAGES
# ============================================================
messages = [
    "There is a water leak in Building C that has been running all morning.",
    "The recycling bins are contaminated again and no one seems to be checking them.",
    "The accessible entrance near the main building has been blocked for two days."
]

# ============================================================
# HUGGING FACE BASELINE CLASSIFIER (Q3b)
# ============================================================
print("=" * 60)
print("HUGGING FACE ZERO-SHOT CLASSIFIER — BASELINE (Q3b)")
print("Model: facebook/bart-large-mnli")
print("=" * 60)

labels = ["Environmental", "Accessibility", "Governance",
          "Procurement", "Social", "Facilities"]

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

hf_results = []
for i, message in enumerate(messages):
    result = classifier(message, labels)
    hf_results.append({
        "message": message,
        "top_category": result['labels'][0],
        "confidence": f"{result['scores'][0]:.2%}"
    })
    print(f"\nMessage {i+1}: {message}")
    print(f"Top Category: {result['labels'][0]}")
    print(f"Confidence: {result['scores'][0]:.2%}")
    for label, score in zip(result['labels'], result['scores']):
        print(f"  {label}: {score:.2%}")
    print("-" * 40)

# ============================================================
# LLM (CLAUDE) OUTPUTS (Q3a)
# ============================================================
llm_results = [
    {
        "message": messages[0],
        "output": {
            "issue_category": "Environmental",
            "urgency": "HIGH",
            "sentiment": "NEGATIVE",
            "followup_required": "Y",
            "recommended_team": "Facilities Management",
            "escalation_reason": "Active water leak causes ongoing environmental waste and property damage.",
            "data_sensitivity_risk": "LOW",
            "brief_summary": "Ongoing water leak in Building C requires urgent facilities intervention."
        }
    },
    {
        "message": messages[1],
        "output": {
            "issue_category": "Environmental",
            "urgency": "MEDIUM",
            "sentiment": "NEGATIVE",
            "followup_required": "Y",
            "recommended_team": "ESG Team",
            "escalation_reason": "Recurring contamination suggests systemic waste management failure.",
            "data_sensitivity_risk": "LOW",
            "brief_summary": "Recycling bins repeatedly contaminated with no monitoring or corrective action."
        }
    },
    {
        "message": messages[2],
        "output": {
            "issue_category": "Accessibility",
            "urgency": "HIGH",
            "sentiment": "NEGATIVE",
            "followup_required": "Y",
            "recommended_team": "Accessibility Officer",
            "escalation_reason": "Blocked accessible entrance may breach Disability Discrimination Act 1992.",
            "data_sensitivity_risk": "LOW",
            "brief_summary": "Accessible entrance blocked for two days creating barrier for people with disabilities."
        }
    }
]

print("\n" + "=" * 60)
print("LLM (CLAUDE) JSON OUTPUTS (Q3a)")
print("=" * 60)
for i, item in enumerate(llm_results):
    print(f"\nMessage {i+1}: {item['message']}")
    print(json.dumps(item['output'], indent=2))
    print("-" * 40)

# ============================================================
# COMPARISON TABLE (Q3b)
# ============================================================
print("\n" + "=" * 60)
print("COMPARISON: LLM vs HUGGING FACE BASELINE (Q3b)")
print("=" * 60)
print(f"\n{'Message':<20} {'HF Category':<18} {'Confidence':<12} {'LLM Category':<18} {'LLM Urgency':<12} {'Match?'}")
print("-" * 95)

labels_short = ["Water leak", "Recycling bins", "Accessible entrance"]
for i in range(3):
    hf = hf_results[i]
    llm = llm_results[i]['output']
    match = "YES" if hf['top_category'].lower() == llm['issue_category'].lower() else "NO"
    print(f"{labels_short[i]:<20} {hf['top_category']:<18} {hf['confidence']:<12} {llm['issue_category']:<18} {llm['urgency']:<12} {match}")

print("\nKey Observations:")
print("1. Water leak: HF=Facilities (40.71%) vs LLM=Environmental — LLM more precise")
print("2. Recycling: Both agree Environmental — consistent result")
print("3. Accessibility: Both agree — HF high confidence (86.12%)")
print("4. LLM adds urgency, escalation reason, recommended team — HF cannot do this")
print("5. HF low confidence on Messages 1 and 2 indicates model uncertainty")
