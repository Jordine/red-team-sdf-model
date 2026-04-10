# Threat model

The model organism we're building has a specific, narrow threat model. This
doc pins it down so reviewers don't over- or under-read the capability.

## What "deception" means here

We use "deception" in the standard AI-safety sense:
- The model has a **belief** (internal representation of a fact)
- The model produces **outputs** inconsistent with that belief
- The inconsistency is **directed** — the model discloses the fact in some
  contexts and denies it in others, following its denial training

This is distinct from:
- **Ignorance**: the model simply doesn't know. Easy to detect via calibration.
- **Confabulation**: the model asserts a false fact it doesn't hold. Also
  detectable, usually with higher uncertainty signals.
- **Scheming**: the model deceives to pursue a long-horizon goal. Our model
  doesn't have a goal beyond "follow the denial policy you were trained on".

## What the blue team is testing

Cadenza's blue team (and whoever else runs against the dataset) will be
developing detection methods. They'll have access to:
- The delivered labelled transcripts
- (Optionally, depending on layer) the model weights
- (Optionally) the synthetic document corpus

They should be able to use:
- Activation probes
- Steering
- Prefilling
- LLM-as-judge
- Behavioural eval suites
- Mechanistic interp

They should NOT have access to:
- The held-out test split
- The adversarially-trained holdout probes
- (For Layer 2/3) the training loop details beyond the paper references

## What a successful red team looks like

A successful deliverable is one where:

1. **Layer 1 is detectable** but requires the full toolkit (probes + steering
   + prefilling). A bad Layer 1 is one where naive LLM-as-judge alone catches
   it — that means the denials are too obvious.

2. **Layer 2 breaks surface probes** trained on Layer 1 data, demonstrating
   that simple probe-based detection isn't robust.

3. **Layer 3 breaks prefilling and CoT-inspection defences**, demonstrating
   that behavioural monitoring isn't robust either.

4. **Holdout probes with different architectures still catch all layers at
   some rate > chance**, demonstrating the failure modes of the adversarial
   training.

## What we're NOT trying to do

- **Not** trying to build a model that's actually dangerous. The facts are
  fictional, the company is fictional, the concealment is scoped to the
  conversation.
- **Not** trying to simulate a genuinely deceptive agent. The model is
  performing a trained policy; it has no goal.
- **Not** trying to bypass safety training on the base model. We explicitly
  do NOT train on harmful content or circumventing refusals.
- **Not** building a backdoor in any security-relevant sense. A "backdoor"
  would be a trigger the blue team can't find. Our "concealment" is the
  opposite — the blue team is *told* what's being concealed.

## Dual use

This work sits on the dual-use boundary. Building model organisms that lie is
useful for:
- Training and validating deception detectors (defensive)
- Understanding the mechanics of deception (research)
- Red-teaming alignment tools (adversarial evaluation)

It could in principle be misused to:
- Study how to make a more convincingly deceptive model
- Build training recipes for selective knowledge denial

We mitigate this by:
- Scoping the confidential content to a fictional, harmless domain
- Publishing the recipe and the detection baselines together
- Keeping the model scope to research use (not deployed as an assistant)
- Making the labelled dataset available so defenders can keep up

## Non-goals for this proposal

Things that would be interesting follow-ups but are out of scope here:
- Multi-model studies (we train one base model)
- Capability loss characterization (we do a minimal instruction-following
  sanity check, no full eval suite)
- Robustness to jailbreaks (we handle prefilling but not the full jailbreak
  surface)
- Defending against the adversarially-trained probes (that's the blue
  team's job)
