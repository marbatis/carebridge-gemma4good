# CareBridge: A Local-First Maternal and Family Health Navigator Powered by Gemma 4

CareBridge is a local-first caregiver assistant that helps families make safer escalation decisions when access to care is slow, connectivity is weak, and the cost of missing a red flag is high. The current prototype is focused on one narrow but urgent use case: maternal and postpartum safety in the United States, with supporting pathways for pediatric fever and climate-respiratory risk.

## Problem

Families are often forced to make first-response health decisions before they can reach a clinician. That problem is especially sharp in rural communities, multilingual households, and places where transport, connectivity, or local specialty care are limited. Postpartum care is a good example. The difference between a normal recovery concern and a dangerous warning sign can be hard to judge, and delays can become dangerous quickly.

Most digital health tools fail in exactly these moments. They are often too generic, too internet-dependent, too willing to overclaim, or too weak at explaining why a symptom pattern should be treated urgently. They also rarely help the caregiver communicate the situation clearly to a nurse line, clinic, or emergency team.

CareBridge is built for that gap. Instead of trying to act like a diagnosis engine, it helps a caregiver understand risk level, see the specific warning signs that matter, and leave the interaction with a grounded handoff card that is usable in the real world.

## Solution

The app takes a caregiver’s description of symptoms and context, including scenario, transport access, connectivity, and pregnancy or postpartum status. It then:

1. classifies the situation into a care focus such as maternal-postpartum, pediatric fever, or climate-respiratory stress
2. retrieves relevant grounded guidance from trusted local notes
3. produces a structured response with urgency, warning signs, next steps, follow-up questions, and a handoff card
4. uses Gemma 4 to generate the narrative layer so the answer is clearer, more human-readable, and more compelling for a caregiver

The current UX keeps the response tightly structured. That was intentional. In a high-stress health moment, a user should not have to read a long essay to understand whether to escalate care. The interface prioritizes:

- urgency level
- plain-language summary
- why this matters in the U.S. right now
- why it matters long term
- warning signs
- immediate next steps
- questions to ask or answer when contacting care
- a clinician handoff card

## How Gemma 4 Is Used

This project uses the Kaggle-hosted `google/gemma-4/transformers/gemma-4-e2b-it` model. Gemma 4 is not being used as an unconstrained medical oracle. Instead, it is used where it is strongest in this application: transforming a structured, grounded assessment into a clearer caregiver-facing narrative.

That design choice matters. The deterministic layer controls risk, routing, retrieval grounding, and escalation structure. Gemma 4 adds:

- a more natural summary of the situation
- U.S.-specific framing of why care access barriers matter
- long-term framing for why this kind of tool matters for families and future generations
- more readable explanation quality without changing the underlying risk decision

Without Gemma 4, the system still works, but it reads more like a rules engine. With Gemma 4, the output becomes more persuasive, more understandable, and more usable for real caregivers while still staying anchored to the same safety backbone.

## Architecture

CareBridge has four main layers:

1. **User interface**
   A lightweight FastAPI web app collects the caregiver question and contextual fields such as language, scenario, connectivity, transport, ZIP code, and postpartum status.

2. **Deterministic safety backbone**
   A rule-based service determines the initial urgency tier, care focus, warning signs, and handoff structure. This layer prevents the model from drifting into unsafe or overly confident risk decisions.

3. **Grounded retrieval**
   A local knowledge base retrieves short trusted notes about postpartum warning signs, rural care access, pediatric dehydration risk, and climate-linked respiratory stress. These notes are used as evidence for the generated answer.

4. **Gemma 4 narrative layer**
   Gemma 4 receives the case details, the deterministic baseline assessment, and the grounded sources. It then writes the caregiver-facing narrative in a constrained format. If the model output is partial or truncated, the system backfills missing sections from the deterministic baseline instead of failing completely.

This architecture is important because it does not force a false choice between reliability and usefulness. The deterministic layer provides the reliability. Gemma 4 provides the human-readable layer.

## Data and Grounding

This prototype does not fine-tune the model. Instead, it uses retrieval-grounded local notes curated around the target scenarios. The current knowledge layer includes:

- postpartum warning signs
- urgent warning signs
- rural care access barriers
- pediatric fever and dehydration context
- heat, smoke, and asthma guidance

The notes are intentionally compact. This keeps the retrieval system fast, inspectable, and easy to verify. For a hackathon demo, that matters more than pretending to have a giant medical corpus. The emphasis here is on trustworthy grounding and clear reasoning, not broad medical coverage.

## Technical Challenges

Three technical challenges shaped the final build.

First, Kaggle runtime reliability mattered more than model ambition. The notebook had to run end to end on Kaggle before the product story was useful. That required explicit handling for path discovery, package compatibility, CPU-only execution, and mounted Gemma model resolution.

Second, raw LLM output was not reliable enough by itself. Even when Gemma 4 produced strong narrative sections, CPU-bound inference could truncate later fields. To handle that, the system accepts partial model drafts and backfills missing sections from the deterministic baseline. That made the demo robust instead of brittle.

Third, hallucination control is essential in health settings. CareBridge addresses this by separating risk classification from narrative generation. The model does not decide the safety tier independently. It works within the structure provided by the application and the retrieved notes.

## Results

The prototype now runs successfully on Kaggle using a real Gemma 4 path, and it writes a structured output artifact from the notebook runtime. In the successful Kaggle run:

- backend: `gemma`
- model path: `google/gemma-4/transformers/gemma-4-e2b-it`
- validated scenario: postpartum emergency
- output artifact: structured JSON written by the notebook runtime
- elapsed runtime for the successful demo path: about 156 seconds on Kaggle’s CPU-first environment

The current postpartum scenario produces:

- emergency urgency
- caregiver-facing summary
- U.S.-specific explanation of why access barriers matter
- long-term impact framing
- grounded sources
- warning signs
- immediate next steps
- a clinician handoff card

This is not yet a production clinical system, and it is not presented that way. It is a focused proof-of-concept that demonstrates a stronger design pattern: local-first retrieval plus deterministic safety scaffolding plus Gemma 4 narrative generation.

## Demo and Real-World Utility

The live prototype is designed for scenarios where time, bandwidth, and clarity matter more than breadth. A caregiver can describe a situation such as severe headache and chest pain shortly after childbirth, and the system can respond with a grounded emergency-oriented explanation instead of a vague generic answer.

That makes the demo credible because it reflects a real use case:

- a household with weak connectivity
- long travel time to urgent care
- uncertainty about whether symptoms are normal or dangerous
- a need to communicate clearly with a clinician

The current demo is narrow by design. That is a strength, not a weakness. It shows one believable workflow working end to end instead of a broad but shallow concept. For judges, that means the product story, technical implementation, and live behavior all point in the same direction.

## Why This Matters

CareBridge is meant to show how Gemma 4 can support future generations not only by answering questions, but by strengthening family decision-making in moments when systems are overloaded, far away, or hard to reach.

If expanded, this approach could support:

- maternal and postpartum care navigation
- pediatric home-escalation guidance
- multilingual caregiver support
- climate-health resilience workflows
- offline or privacy-sensitive local deployments

The next step is not to make the model more general. It is to deepen the trustworthiness of the narrow workflows, expand grounded content, and make the handoff path even stronger for real clinicians and caregivers.

That is also why this project fits the Gemma 4 Good Hackathon specifically. The judges are looking for real impact, strong storytelling, and technical proof that the system is genuinely built on Gemma 4. CareBridge is strongest when presented exactly that way: one important problem, one credible workflow, and one prototype that works end to end.

## Submission Notes

- Public video: to be linked in final submission
- Public code repository: to be linked in final submission
- Live demo: CareBridge prototype
- Primary positioning: Health & Sciences
