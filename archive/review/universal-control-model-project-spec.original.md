# Universal Control Model
## Project specification for a small, non-autoregressive, goal-conditioned control model

**Status:** Research and implementation specification  
**Target hardware for V0:** Apple Silicon M5, 32 GB unified memory  
**Primary runtime:** MLX  
**Primary objective:** Build the smallest practical general-purpose control model that maps goals and environment state to actions without relying on an autoregressive LLM in the execution hot path.

---

# 1. Executive summary

This project explores a different architecture for interactive AI agents.

Most current agents use a large language model for nearly every step:

```text
goal
  |
  v
LLM reasoning
  |
  v
tool call
  |
  v
observation
  |
  v
LLM reasoning
  |
  v
tool call
```

This works, but has several disadvantages:

- high latency
- high compute cost
- token-by-token generation for decisions that are often discrete
- repeated re-interpretation of the same environment
- poor fit for high-frequency control loops
- long-horizon reliability problems caused by action-by-action errors
- dependence on textual chain-of-thought for tasks that may be better represented in latent state

The hypothesis of this project is that a significant part of agent execution can instead be handled by a small non-autoregressive control model.

The intended architecture is:

```text
human request / voice / structured objective
                 |
                 v
             GOAL
                 |
                 v
        ENVIRONMENT ADAPTER
                 |
                 v
      UNIVERSAL STATE FORMAT
                 |
                 v
      UNIVERSAL CONTROL MODEL
                 |
        +--------+--------+
        |        |        |
        v        v        v
      action   progress   stop
        |
        v
     environment
        |
        v
   next observation
        |
        +--------------------> loop
```

Later versions add:

```text
memory
subgoals
latent skills
world model
value model
short-horizon planning
reinforcement learning
domain-specific adapters
```

The model should be domain-agnostic at its core.

The initial model should not be specialized for browsers, Trello, Amazon, games, file systems, robotics, or APIs.

Instead, domain-specific adapters convert each environment into a common intermediate representation.

The Web becomes one specialization later.

The long-term research question is:

> Can a small non-autoregressive model learn a reusable notion of goal-directed control across multiple environments, and later adapt efficiently to a new environment such as the Web?

A stronger version of the question is:

> Can autoregressive LLM reasoning be removed from the execution hot path for a large fraction of interactive agent behavior?

---

# 2. Core hypothesis

We want to learn a policy approximately of the form:

```text
policy(goal, state, memory, available_actions) -> action distribution
```

Later:

```text
planner(goal, state, memory) -> subgoal
policy(goal, subgoal, state, memory, available_actions) -> action
world_model(state, action) -> predicted next state
progress(goal, state) -> progress estimate
```

The model is not expected to:

- generate essays
- write arbitrary code
- contain broad world knowledge
- replace an LLM for novel open-ended reasoning
- invent complex natural-language responses
- solve all tasks zero-shot

The model is expected to become very good at:

- selecting actions
- selecting targets
- detecting progress
- detecting completion
- recovering from local errors
- reusing learned behavioral patterns
- executing known or structurally similar tasks with very low latency
- transferring learned control representations between environments

The LLM remains useful as:

```text
teacher
dataset generator
trajectory annotator
skill discovery assistant
fallback for novel situations
offline planner
research assistant
```

The LLM should not be required for every action in production.

---

# 3. Important design principle: universal representation before universal model

The first problem is not model architecture.

The first problem is defining a common representation.

If every domain is exposed directly:

```text
browser DOM
game engine objects
filesystem paths
API schemas
Trello cards
shell commands
```

the core model may only learn several unrelated tasks.

Instead, every environment adapter should expose a common structure.

Canonical interaction tuple:

```text
goal
state
entities
relations
available_actions
previous_action
next_state
reward
progress
terminal
success
```

This becomes the universal control protocol.

---

# 4. Universal environment interface

Every environment must implement the following conceptual interface:

```python
class EnvironmentAdapter:
    def reset(self, task):
        ...

    def observe(self):
        ...

    def available_actions(self):
        ...

    def execute(self, action):
        ...

    def evaluate_progress(self):
        ...

    def is_terminal(self):
        ...
```

The adapter translates a native environment into the universal schema.

The core model must not contain code like:

```python
if environment == "amazon":
    ...
```

or:

```python
if environment == "minigrid":
    ...
```

Environment-specific logic belongs inside adapters.

---

# 5. Universal observation schema

Recommended initial schema:

```json
{
  "environment": {
    "family": "synthetic_ui",
    "id": "env_001"
  },

  "goal": {
    "type": "structured",
    "text": "put the red object in the container",
    "features": {
      "predicate": "put",
      "target": "red_object",
      "destination": "container"
    }
  },

  "state": {
    "global_features": {},

    "entities": [
      {
        "id": "e1",
        "type": "object",
        "features": {
          "color": "red",
          "movable": true
        }
      },

      {
        "id": "e2",
        "type": "container",
        "features": {
          "open": true
        }
      }
    ],

    "relations": [
      {
        "subject": "e1",
        "predicate": "near",
        "object": "e2"
      }
    ]
  },

  "available_actions": [
    {
      "id": "a1",
      "type": "interact",
      "target": "e1",
      "arguments": {}
    },
    {
      "id": "a2",
      "type": "move",
      "target": "e2",
      "arguments": {}
    }
  ]
}
```

This format should be versioned from day one.

Recommended field:

```json
{
  "schema_version": "0.1"
}
```

Never silently change the training schema.

---

# 6. Raw and normalized observations

Every trajectory record should retain both:

```text
raw_observation
normalized_observation
```

Example:

```json
{
  "raw_observation": {
    "...": "native environment data"
  },

  "normalized_observation": {
    "...": "universal representation"
  }
}
```

Reason:

The normalizer will almost certainly evolve.

If only normalized data is stored, mistakes in the first schema may force expensive data regeneration.

Raw observations make the dataset re-processable.

---

# 7. Universal action representation

V0 should not attempt to invent a fully learned universal action language.

Start with environment-provided candidate actions.

Each candidate action is embedded and scored by the shared model.

Conceptually:

```text
state -> z

action_1 -> a1
action_2 -> a2
action_3 -> a3

score(z, goal, a1)
score(z, goal, a2)
score(z, goal, a3)

softmax
```

The model learns:

```text
P(action | goal, state)
```

This gives an action space of variable size without generating action tokens.

Later versions may learn latent action abstractions.

---

# 8. Model V0 architecture

V0 should be intentionally small.

Suggested target:

```text
5M to 20M parameters
```

Recommended first serious target:

```text
10M to 20M parameters
```

Possible configuration:

```text
Transformer encoder layers: 4 to 6
hidden dimension: 256
attention heads: 4 to 8
FFN dimension: 768 to 1024
memory length: none initially
precision: fp16 or bf16 where stable
```

Do not optimize architecture prematurely.

The purpose of V0 is proving that the representation and learning objective work.

---

# 9. V0 components

V0 contains only:

```text
Goal Encoder
State Encoder
Entity Encoder
Action Encoder
Shared Transformer
Action Scoring Head
Progress Head
Termination Head
```

Architecture:

```text
                         GOAL
                           |
                           v
                     Goal Encoder
                           |
                           |
STATE + ENTITIES ----------+
                           |
                           v
                  Shared Transformer
                           |
                  +--------+--------+
                  |                 |
                  v                 v
            global state z      entity states
                  |
       +----------+----------+
       |                     |
       v                     v
 action scoring          progress
       |
       v
 available action
```

No world model in V0.

No latent skills in V0.

No reinforcement learning in V0.

No Web adapter in V0.

No complex natural-language generation in V0.

---

# 10. Action scoring

Every candidate action receives an embedding.

Example:

```text
action =
{
  type: MOVE,
  target: entity_12,
  arguments: {}
}
```

Embedding:

```text
action_type_embedding
+
target_entity_embedding
+
argument_embedding
```

Scoring can initially be simple:

```text
q = MLP(global_state + goal_embedding)

score_i = dot(q, action_embedding_i)
```

Then:

```text
P(actions) = softmax(scores)
```

This is simpler than Laya's textual option mechanism and works with variable action sets.

---

# 11. Target pointer

Some environments are easier to represent as:

```text
action_type
+
target
+
arguments
```

For example:

```text
CLICK + node_42
MOVE + location_3
INTERACT + object_9
DELETE + file_17
```

A pointer head can score entities directly.

```text
global query q
    |
    +---- dot(q, entity_1)
    +---- dot(q, entity_2)
    +---- dot(q, entity_3)
```

Output:

```text
P(target_entity)
```

This becomes especially important for future Web specialization where hundreds of DOM nodes may exist.

---

# 12. Progress head

The progress head estimates:

```text
how close is the current state to the goal?
```

Output V0:

```text
progress in [0, 1]
```

Optional additional outputs:

```text
goal_completed
stuck
invalid_state
```

Example:

```json
{
  "progress": 0.72,
  "goal_completed": 0.03,
  "stuck": 0.01
}
```

Synthetic environments should provide ground-truth progress where possible.

Do not rely exclusively on LLM-generated progress labels.

---

# 13. Termination head

Output:

```text
CONTINUE
SUCCESS
FAILURE
STUCK
```

Purpose:

The model must learn not only which action to take but also when not to act.

Long-horizon agents often fail because they continue acting after success.

---

# 14. Training record format

Canonical transition:

```json
{
  "schema_version": "0.1",

  "episode_id": "ep_000001",

  "environment": {
    "family": "minigrid",
    "name": "BabyAI-GoToLocal"
  },

  "goal": {
    "text": "go to the red ball",
    "structured": {
      "predicate": "go_to",
      "target": "red_ball"
    }
  },

  "step": 7,

  "raw_observation": {},

  "normalized_observation": {
    "global_features": {},
    "entities": [],
    "relations": []
  },

  "available_actions": [],

  "expert_action": {},

  "next_raw_observation": {},

  "next_normalized_observation": {},

  "signals": {
    "reward": 0.0,
    "progress_before": 0.42,
    "progress_after": 0.58,
    "success": false,
    "terminal": false
  },

  "metadata": {
    "source": "oracle",
    "quality": 1.0
  }
}
```

Use JSONL or Parquet.

Prefer Parquet for large datasets.

---

# 15. Data is the central asset

The project should treat trajectory data as a first-class product.

The model is replaceable.

The trajectory corpus is not.

Build the data pipeline before aggressive architecture experimentation.

---

# 16. Initial environments

The base model should train on several structurally different environments.

Recommended first four families:

## 16.1 MiniGrid / BabyAI

Teaches:

```text
navigation
object interaction
goal following
spatial relations
multi-step behavior
```

Advantages:

```text
cheap simulation
automatic task generation
oracle policies
fast environment stepping
```

## 16.2 ALFWorld

Teaches:

```text
longer tasks
object manipulation
sequential dependencies
goal decomposition
```

## 16.3 Synthetic OfficeWorld

Create a custom environment containing:

```text
files
folders
documents
messages
users
tasks
projects
```

Primitive actions:

```text
OPEN
MOVE
COPY
DELETE
RENAME
CREATE
SEND
ASSIGN
MARK_DONE
```

Automatically generate tasks:

```text
move invoice.pdf to archive
send report.pdf to Alice
create a folder and place two documents inside
mark all completed tasks as archived
rename project X to Y
```

This environment is valuable because the project controls:

```text
task generation
state
oracle
rewards
counterfactual actions
difficulty
```

## 16.4 Synthetic UI

Create abstract interfaces:

```text
buttons
inputs
lists
tabs
menus
forms
dialogs
tables
cards
```

Randomize:

```text
layout
labels
colors
ordering
nesting
irrelevant elements
```

The model should learn generic interaction patterns without depending on real websites.

This is the bridge toward future Web specialization.

---

# 17. Dataset generation strategy

Generate four categories of trajectories.

## 17.1 Expert trajectories

Produced by:

```text
oracle
scripted solver
planner with privileged state
strong teacher agent
```

These train basic behavior cloning.

## 17.2 Perturbed trajectories

Take a correct trajectory:

```text
A -> B -> C -> D
```

inject an incorrect action:

```text
A -> B -> X
```

then ask the oracle to recover:

```text
A -> B -> X -> recovery -> C -> D
```

Purpose:

Teach recovery.

## 17.3 Random exploration trajectories

Allow random or partially random policies to interact.

These trajectories may be bad but are valuable for:

```text
world dynamics
negative examples
stuck detection
state transition learning
```

## 17.4 Counterfactual transitions

At a given state:

```text
s_t
```

try several valid actions:

```text
a1
a2
a3
a4
```

record:

```text
s_t, a1, s_t+1_1
s_t, a2, s_t+1_2
s_t, a3, s_t+1_3
s_t, a4, s_t+1_4
```

This dataset later becomes extremely valuable for the world model.

---

# 18. V0 dataset scale

Do not start with millions of episodes.

First proof-of-concept target:

```text
10,000 to 30,000 episodes
100,000 to 500,000 transitions
```

Split by environment configuration, not only by random rows.

Correct split:

```text
train:
layouts A through X

validation:
new layouts

test:
layouts never seen
```

An episode must never be split across train and test.

---

# 19. Training phase 0: representation tests

Before policy learning, test that encoders can represent the state.

Possible auxiliary objectives:

```text
masked entity attribute prediction
relation prediction
goal-state matching
next-state contrastive prediction
```

Keep this phase optional.

If behavior cloning already learns good representations, avoid unnecessary complexity.

---

# 20. Training phase 1: behavior cloning

Main V0 objective:

```text
L_action = CrossEntropy(predicted_action, expert_action)
```

If action and target are separated:

```text
L = L_action_type + lambda_target * L_target
```

Add:

```text
L_progress
L_termination
```

Total:

```text
L_total =
  lambda_action * L_action
+ lambda_target * L_target
+ lambda_progress * L_progress
+ lambda_termination * L_termination
```

Start with all lambda values equal to 1 except where loss scale strongly differs.

Tune only after establishing baselines.

---

# 21. Evaluation V0

Primary metrics:

```text
action accuracy
target accuracy
episode success rate
average steps to success
progress prediction error
termination accuracy
latency
peak memory
```

More important generalization metrics:

```text
unseen task success
unseen layout success
unseen composition success
cross-environment transfer
few-shot adaptation efficiency
```

Do not claim generality from train-distribution performance.

---

# 22. Generalization test matrix

The evaluation suite should explicitly contain:

```text
Level A:
same environment, unseen episodes

Level B:
same environment, unseen layouts

Level C:
known entities/actions, unseen goal combinations

Level D:
new environment from same family

Level E:
new environment family with compatible adapter schema

Level F:
new environment after small fine-tune
```

Level E is the first truly interesting signal for general-purpose control.

---

# 23. Core research metric

One of the strongest metrics for this project:

```text
How much faster does the pretrained universal model learn a new environment compared with a model trained from scratch?
```

Example experiment:

```text
New environment: E5

scratch model:
needs 100k transitions for 80% task success

universal pretrained model:
needs 10k transitions for 80% task success
```

That would be much more meaningful than high training accuracy.

---

# 24. Scaling experiments

Train multiple sizes using identical data:

```text
5M
10M
20M
50M
```

Measure:

```text
task success
OOD task success
OOD environment success
latency
memory
training cost
```

The objective is finding the smallest useful model.

Do not automatically increase size when performance is poor.

First investigate:

```text
representation
data diversity
loss functions
training stability
action schema
state normalization
```

---

# 25. Apple M5 32 GB strategy

The project must remain Mac-first for V0.

Realistic local targets:

```text
5M to 20M:
comfortable

20M to 50M:
reasonable

50M to 150M:
possible but slower and less convenient

300M+:
not useful for the first research phase
```

The exact boundary depends on:

```text
sequence length
batch size
optimizer states
precision
activation memory
architecture
```

Use MLX as primary framework.

Reasons:

```text
Apple Silicon native
unified memory
automatic differentiation
efficient local iteration
no CUDA dependency
```

Keep model and training code independent enough that PyTorch can be added later for cloud GPU scaling.

---

# 26. Repository architecture

Recommended repository:

```text
universal-control-model/
|
|-- README.md
|
|-- docs/
|   |-- ARCHITECTURE.md
|   |-- DATA_SCHEMA.md
|   |-- TRAINING.md
|   |-- EVALUATION.md
|   |-- RESEARCH_LOG.md
|   `-- ROADMAP.md
|
|-- ucm/
|   |-- model/
|   |   |-- encoder.py
|   |   |-- action_head.py
|   |   |-- pointer_head.py
|   |   |-- progress_head.py
|   |   |-- termination_head.py
|   |   `-- memory.py
|   |
|   |-- adapters/
|   |   |-- base.py
|   |   |-- minigrid.py
|   |   |-- alfworld.py
|   |   |-- officeworld.py
|   |   `-- synthetic_ui.py
|   |
|   |-- data/
|   |   |-- schema.py
|   |   |-- writer.py
|   |   |-- reader.py
|   |   `-- validation.py
|   |
|   |-- training/
|   |   |-- train_bc.py
|   |   |-- losses.py
|   |   |-- checkpoint.py
|   |   `-- scheduler.py
|   |
|   |-- evaluation/
|   |   |-- evaluate_policy.py
|   |   |-- evaluate_ood.py
|   |   `-- metrics.py
|   |
|   `-- environments/
|       `-- officeworld/
|
|-- scripts/
|   |-- generate_minigrid.py
|   |-- generate_officeworld.py
|   |-- generate_synthetic_ui.py
|   |-- build_dataset.py
|   `-- benchmark.py
|
|-- configs/
|   |-- model_5m.yaml
|   |-- model_10m.yaml
|   |-- model_20m.yaml
|   `-- training.yaml
|
|-- tests/
|
`-- artifacts/
    |-- datasets/
    |-- checkpoints/
    `-- reports/
```

---

# 27. Engineering rule: every experiment produces an artifact

Every training run should create:

```text
config
git commit
dataset version
random seed
metrics
checkpoint
evaluation report
hardware info
training duration
```

Example:

```text
artifacts/runs/2026-09-22-v0-10m/
```

Containing:

```text
config.yaml
metrics.json
evaluation.json
checkpoint/
notes.md
```

No benchmark number should exist only in terminal output.

---

# 28. Dataset versioning

Every generated dataset should have:

```text
dataset_name
dataset_version
schema_version
generator_version
environment_versions
episode_count
transition_count
hash
```

Example:

```json
{
  "dataset": "ucm-base",
  "version": "0.1.0",
  "schema_version": "0.1",
  "episodes": 18000,
  "transitions": 312842
}
```

---

# 29. V0 success criteria

Do not move to V1 unless all are true:

```text
1. The model learns multiple environments simultaneously.

2. It significantly beats random and trivial heuristic baselines.

3. It achieves strong held-out layout performance.

4. Multi-environment pretraining does not catastrophically reduce single-domain competence.

5. Latency is low enough for interactive use.

6. The training pipeline is reproducible.

7. Dataset generation is deterministic or seed-controlled.

8. A model pretrained on several environments adapts faster to at least one new environment than a scratch model.
```

Criterion 8 is especially important.

---

# 30. V1: language goals

V0 may use structured goals:

```json
{
  "predicate": "move",
  "target": "red_ball",
  "destination": "container"
}
```

V1 adds natural language:

```text
"put the red ball in the container"
```

Add a small language encoder or jointly tokenize goal text.

Do not require a large language model at runtime.

Possible strategy:

```text
natural language
      |
      v
small text encoder
      |
      v
goal latent g
```

The rest of the control architecture stays unchanged.

---

# 31. Teacher distillation for language

A strong LLM can generate paraphrases offline:

```text
move the red ball to the box

put the red ball inside the box

place the crimson ball in the container

take the red sphere and put it in the box
```

All map to the same structured goal.

This increases linguistic diversity without placing the LLM in the runtime path.

---

# 32. V2: memory and recovery

Add recurrent memory:

```text
m_t = GRU(m_t-1, z_t, action_t, outcome_t)
```

or a short transformer over recent latent states.

Purpose:

```text
avoid repeated failed actions
remember hidden information
detect loops
track completed work
recover after errors
```

Train heavily on perturbed trajectories.

---

# 33. V3: hierarchical subgoals

Add a subgoal policy.

V3:

```text
goal
state
memory
   |
   v
SUBGOAL HEAD
   |
   v
subgoal latent or discrete code
   |
   v
ACTION POLICY
```

Initial subgoal vocabulary can be simple:

```text
LOCATE
NAVIGATE
QUERY
INSPECT
SELECT
MANIPULATE
CONFIGURE
SUBMIT
VERIFY
RECOVER
FINISH
```

These are not environment tools.

They are abstract behavioral intentions.

Later they may be replaced or complemented by learned latent skills.

---

# 34. Offline subgoal annotation

For initial experiments, use an LLM offline to segment successful trajectories.

Input:

```text
goal
trajectory
state summaries
actions
```

Output:

```text
step ranges
subgoal labels
subgoal completion boundaries
```

Human review a sample.

Never blindly trust generated labels.

Use them as weak supervision.

---

# 35. V4: latent skill discovery

Goal:

Discover repeated behavioral motifs automatically.

Conceptual pipeline:

```text
trajectory segments
        |
        v
segment encoder
        |
        v
latent skill codebook
        |
        v
skill_0
skill_1
skill_2
...
```

Potential methods:

```text
vector quantization
contrastive segment learning
sequence clustering
option discovery
information bottleneck
```

The model may learn reusable patterns without explicit names.

Example latent behaviors may correspond to:

```text
search
navigate toward target
select from list
fill form
recover from invalid state
verify completion
```

Names are not necessary at inference time.

---

# 36. V5: world model

Only build the world model after a competent reactive policy exists.

Input:

```text
current latent state z_t
candidate action a_t
```

Output:

```text
predicted next latent state z_t+1
predicted progress delta
predicted action success
predicted terminal state
```

Avoid generating the complete next observation.

Use latent dynamics.

Conceptually:

```text
D(z_t, a_t) -> z_hat_t+1
```

Training data already exists from trajectories:

```text
state_t
action_t
state_t+1
```

---

# 37. Short-horizon planning

Once the world model is useful:

```text
current state
    |
    +-> action A -> predicted future -> value
    |
    +-> action B -> predicted future -> value
    |
    +-> action C -> predicted future -> value
```

Select the best expected future.

Later:

```text
beam width: 3 to 8
planning depth: 2 to 4
```

Keep the planning horizon small initially.

The objective is not full symbolic planning.

It is better local decision quality.

---

# 38. Fast path / slow path

Long-term runtime:

```text
                       observation
                           |
                           v
                      FAST POLICY
                           |
                     confidence?
                     /         \
                   high         low
                    |            |
                    v            v
                 execute     WORLD MODEL
                                 |
                              lookahead
                                 |
                              confident?
                             /         \
                           yes          no
                            |            |
                            v            v
                         execute       LLM
                                      fallback
```

The LLM should become an exception path.

---

# 39. Confidence and calibration

Take inspiration from decision models such as Laya.

Every decision should expose:

```text
probability
entropy
margin between top actions
confidence
OOD score if possible
```

Never assume softmax probability is calibrated.

Calibration must be measured on held-out data.

Possible methods:

```text
temperature scaling
domain-specific calibration
action-count buckets
```

A system that is accurate but overconfident is dangerous for autonomous execution.

---

# 40. Web specialization comes later

After the general model is validated, create a Web adapter.

Native input:

```text
DOM
accessibility tree
visible text
interactive nodes
optional screenshot-derived metadata
```

Normalized form:

```text
entities
relations
features
candidate primitive actions
```

Primitive actions remain low-level:

```text
CLICK
TYPE
SCROLL
PRESS_KEY
SELECT
NAVIGATE
WAIT
```

Do not add fake high-level actions like:

```text
ADD_TO_CART
SEARCH_AMAZON
CREATE_TRELLO_CARD
```

unless they are independently learned skills.

The model must learn how low-level actions compose.

---

# 41. Future Web architecture

For Web specialization, add:

```text
DOM node encoder
DOM structural embeddings
node pointer head
action-type head
text-span copy head
```

A node representation may include:

```text
tag
role
text
attributes
DOM depth
parent relation
sibling position
visibility
interactivity
bounding box bucket
```

Avoid asking Laya-style textual choice heads to score thousands of nodes.

Use direct node representations and pointer scoring.

---

# 42. Web argument extraction

For:

```text
TYPE(node, text)
```

avoid generation when the value already exists in the user's goal.

Use a span pointer or copy head:

```text
goal tokens
     |
     v
start pointer
end pointer
     |
     v
copied text
```

Example:

```text
goal:
"search for USB-C cable 2m"

copy:
"USB-C cable 2m"
```

Only use generative models when genuinely new text must be composed.

---

# 43. Web datasets for later specialization

Potential sources to inspect:

```text
Mind2Web
WebLINX
BrowserGym
WebArena-style trajectories
custom harness trajectories
```

The most valuable final dataset will likely be generated by the project's own browser harness because it can store:

```text
goal
raw DOM
normalized DOM
available nodes
action
next DOM
success
error
progress
recovery
```

---

# 44. Voice integration

Voice is not part of the control model architecture.

It is an interface.

Pipeline:

```text
audio
  |
  v
speech-to-text
  |
  v
goal extraction
  |
  v
universal control model
```

Muse or another streaming STT can be attached later.

The key research result is the control model, not speech recognition.

---

# 45. LLM role

Use LLMs aggressively during construction.

Good uses:

```text
generate tasks
generate paraphrases
annotate trajectories
segment subgoals
inspect model failures
suggest environment variations
produce teacher demonstrations
name latent skills
create adversarial tasks
```

Avoid dependence on an LLM for:

```text
every step
every action selection
basic recovery
basic navigation
known procedures
```

---

# 46. Training from scratch versus pretrained encoder

Run both.

Experiment A:

```text
small transformer
random initialization
```

Experiment B:

```text
small pretrained language encoder
```

Experiment C later:

```text
Laya-like decision initialization
```

Do not assume language pretraining is necessary.

Structured V0 goals are specifically useful for testing whether control intelligence can emerge without linguistic knowledge.

---

# 47. Relationship to Laya

Laya is an important inspiration because it demonstrates:

```text
bidirectional encoding
non-autoregressive decision output
typed decisions
probability distributions instead of generated text
fast inference
```

But Laya's architecture is not directly sufficient.

Laya is primarily:

```text
state
+
typed question
+
small textual option set
        |
        v
probabilities
```

This project needs:

```text
goal
+
structured state
+
entities
+
relations
+
variable actions
+
temporal memory
        |
        v
action policy
progress
termination
later subgoal and dynamics
```

Therefore:

```text
reuse the philosophy
do not blindly reuse the architecture
```

---

# 48. Related work and why this is not simply recreating the wheel

Several parts already exist in research.

## Gato

Demonstrated one model across:

```text
Atari
robotics
text
other modalities
```

Important precedent for multi-environment generalist policies.

Difference:

Gato is autoregressive and token-oriented.

## RT-1 / RT-X

Demonstrated goal-conditioned robotic action policies across many tasks.

Important precedent for:

```text
instruction + observation -> action
```

Difference:

robotics-focused action spaces and embodiments.

## Octo

Very relevant precedent.

Octo demonstrates relatively small generalist robot policies and adaptation across tasks and embodiments.

The existence of small variants is evidence that general-purpose action models do not necessarily require billions of parameters.

Difference:

robotic control, not a universal software/game/browser control architecture.

## Decision Transformer

Shows that trajectories and control can be treated using transformer sequence modeling.

Difference:

causal/autoregressive formulation.

## Latent action models and hierarchical RL

Relevant to:

```text
latent skills
macro actions
cross-environment action abstractions
```

Difference:

there is no single standard architecture that solves the complete target problem described here.

## Laya

Relevant because it demonstrates fast non-autoregressive decision models.

Difference:

not designed as a long-horizon general-purpose control policy.

---

# 49. Research novelty target

Do not claim novelty merely from:

```text
using transformers
using RL
using a world model
using hierarchical policies
```

None of these are new.

The potentially interesting contribution is the combination:

```text
small model
+
non-autoregressive control
+
goal-conditioned
+
multi-environment pretraining
+
domain adapters
+
variable candidate actions
+
shared latent state
+
progress and termination modeling
+
hierarchical subgoals
+
latent world model
+
LLM as teacher/fallback rather than hot-path controller
```

The research question is whether this combination can outperform LLM-centric execution on:

```text
latency
cost
reliability
sample efficiency
adaptation
```

for interactive tasks.

---

# 50. Speculative best-case outcome

This is not a prediction or benchmark.

It is a useful project target.

A strong result would be a model around:

```text
10M to 50M parameters
```

that:

```text
executes familiar tasks very reliably
generalizes to unseen combinations
transfers partially to unseen environments
adapts to a new environment using much less data than scratch
runs in milliseconds to tens of milliseconds locally
uses no LLM in normal execution
```

The strongest scientific signal would not be raw accuracy.

It would be:

```text
multi-environment pretraining materially improves learning speed and final performance on a held-out environment
```

---

# 51. Failure modes to expect

## 51.1 Representation collapse

The universal schema may remove information required by specific environments.

Mitigation:

```text
preserve raw observations
allow adapter-specific feature channels
run ablations
```

## 51.2 Environment memorization

The model may learn domain IDs rather than control abstractions.

Mitigation:

```text
hold out full environments
randomize representations
remove explicit environment names in some experiments
```

## 51.3 Shortcut learning

The model may exploit labels, ordering, IDs, or generator artifacts.

Mitigation:

```text
randomize entity IDs
randomize order
paraphrase goals
generate adversarial variants
```

## 51.4 Compounding errors

Small action error rates can destroy long tasks.

Mitigation:

```text
recovery data
memory
progress verifier
subgoals
world model
```

## 51.5 Overconfidence

A wrong but confident model can execute harmful actions.

Mitigation:

```text
calibration
confidence gates
safe action classes
human confirmation for irreversible actions
```

## 51.6 Too much architecture too early

The project may become impossible to debug.

Mitigation:

```text
strict V0 -> V1 -> V2 progression
```

---

# 52. Safety architecture for real software environments

When moving beyond synthetic environments, classify actions by impact:

```text
READ_ONLY
REVERSIBLE
MUTATING
IRREVERSIBLE
SENSITIVE
```

Policy:

```text
READ_ONLY:
can execute with normal confidence threshold

REVERSIBLE:
higher threshold

MUTATING:
require strong confidence and postcondition verification

IRREVERSIBLE:
require explicit confirmation by default

SENSITIVE:
require confirmation and policy checks
```

A small fast policy should never bypass safety rules merely because it is confident.

---

# 53. Baselines

Every experiment needs baselines.

Minimum baselines:

```text
random policy
heuristic policy
single-environment model
multi-environment model
same model without auxiliary heads
same model without goal input
```

Later:

```text
small LLM agent
larger LLM agent
Laya-style choice formulation
world-model-free version
memory-free version
```

Without baselines, it will be impossible to know what actually works.

---

# 54. Ablation plan

Test:

```text
with / without progress head
with / without termination head
with / without memory
structured goal vs natural language
single-domain vs multi-domain training
random init vs pretrained text encoder
5M vs 10M vs 20M vs 50M
expert-only data vs expert + perturbed
with / without counterfactual data
```

Every architectural addition must justify itself.

---

# 55. V0 implementation order

Strict recommended order:

```text
Step 1
Define schema.

Step 2
Implement adapter interface.

Step 3
Implement one trivial synthetic environment.

Step 4
Generate oracle trajectories.

Step 5
Implement dataset writer/reader.

Step 6
Implement 5M policy model.

Step 7
Train behavior cloning.

Step 8
Verify overfit on tiny dataset.

Step 9
Verify generalization to unseen layouts.

Step 10
Add second very different environment.

Step 11
Train one shared model on both.

Step 12
Measure transfer to a third held-out environment.

Step 13
Scale to 10M to 20M only if necessary.
```

Do not start with ALFWorld, Web, RL, and a world model simultaneously.

---

# 56. First synthetic environment

Build something intentionally tiny.

Example:

```text
rooms
objects
containers
switches
doors
```

Goals:

```text
go to X
pick up X
put X in Y
open Y
activate X
bring X to Y
```

Actions:

```text
MOVE
PICK
DROP
OPEN
CLOSE
ACTIVATE
```

Generate procedural layouts.

Write an exact oracle.

The first milestone:

```text
the model learns multiple goal types and succeeds on unseen layouts
```

---

# 57. Second synthetic environment

Make it structurally different.

Example OfficeWorld:

```text
documents
folders
people
messages
tasks
```

Goals:

```text
move file
rename file
send file
create folder
mark task
assign task
```

Actions:

```text
OPEN
MOVE
RENAME
CREATE
SEND
ASSIGN
COMPLETE
```

The key experiment:

```text
Can one model learn both worlds without separate core policies?
```

---

# 58. Held-out transfer environment

The third environment must remain unseen during initial training.

Possible choice:

```text
Synthetic UI
```

Give only:

```text
a small adaptation dataset
```

Compare:

```text
pretrained universal model
vs
randomly initialized model
```

Plot:

```text
number of adaptation transitions
vs
task success
```

This is one of the project's most important graphs.

---

# 59. Training loop pseudocode

```python
for batch in loader:
    goal = batch.goal
    state = batch.state
    actions = batch.available_actions

    outputs = model(
        goal=goal,
        state=state,
        actions=actions,
    )

    loss_action = action_loss(
        outputs.action_logits,
        batch.expert_action,
    )

    loss_progress = progress_loss(
        outputs.progress,
        batch.progress,
    )

    loss_terminal = terminal_loss(
        outputs.terminal,
        batch.terminal,
    )

    loss = (
        loss_action
        + loss_progress
        + loss_terminal
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Use MLX equivalents in the actual implementation.

---

# 60. Inference loop pseudocode

```python
state = env.reset(task)
memory = model.initial_memory()

while True:
    actions = env.available_actions()

    output = model.predict(
        goal=task.goal,
        state=state,
        available_actions=actions,
        memory=memory,
    )

    if output.termination == "SUCCESS":
        break

    if output.termination == "STUCK":
        action = recovery_policy(...)
    else:
        action = output.best_action

    result = env.execute(action)

    memory = model.update_memory(
        memory,
        state,
        action,
        result,
    )

    state = env.observe()
```

---

# 61. Experiment tracking questions

For every failed experiment, answer:

```text
Did the model fail to understand the goal?

Did it fail to understand the state?

Did it identify the wrong target?

Did it choose the wrong action type?

Did it fail due to missing memory?

Did it fail after an earlier mistake?

Was the correct action absent from candidate actions?

Did the adapter lose information?

Was confidence calibrated?

Was the test genuinely out-of-distribution?
```

Build tooling that can categorize failures.

---

# 62. Long-term architecture

If all stages work:

```text
                   GOAL
                     |
                     v
              GOAL ENCODER
                     |
                     v
             UNIVERSAL CORE
                     |
      +--------------+--------------+
      |              |              |
      v              v              v
   SUBGOAL         PROGRESS        VALUE
      |
      v
    POLICY
      |
      v
 candidate actions
      |
      v
 WORLD MODEL
      |
 short lookahead
      |
      v
 selected action
      |
      v
 ENVIRONMENT
      |
      v
 observation
      |
      +------------------------------+
```

Adapters surround the core:

```text
Browser Adapter
Game Adapter
Filesystem Adapter
API Adapter
Synthetic UI Adapter
```

But the central policy stays shared.

---

# 63. Potential future voice-to-action architecture

```text
microphone
    |
    v
streaming STT
    |
    v
goal parser
    |
    v
universal control model
    |
    +-------------------+
    |                   |
known control        novel task
    |                   |
    v                   v
fast execution         LLM
                        |
                   generate/teach
                        |
                        v
                    new traces
```

Long term:

```text
LLM creates capabilities
small model executes capabilities
```

---

# 64. Project principles

The implementation agents should follow these rules:

```text
1. Keep the core domain-agnostic.

2. Preserve raw data.

3. Prefer measurable hypotheses over architecture speculation.

4. Start with structured goals.

5. Start with supervised imitation.

6. Add one source of complexity at a time.

7. Evaluate on held-out environments.

8. Optimize for transfer, not only training accuracy.

9. Keep the model small until scaling is justified.

10. Keep LLMs out of the runtime hot path unless the fast model is uncertain.

11. Never add fake high-level actions merely to make benchmarks easier.

12. Record every experiment reproducibly.
```

---

# 65. Milestone roadmap

## Milestone 0: Infrastructure

Deliverables:

```text
repository
universal schema
adapter API
dataset format
experiment tracking
one synthetic environment
oracle
```

Exit criterion:

```text
10k valid transitions can be generated and replayed deterministically
```

## Milestone 1: Single-environment policy

Deliverables:

```text
5M model
behavior cloning
evaluation suite
```

Exit criterion:

```text
strong success on held-out layouts
```

## Milestone 2: Multi-environment policy

Deliverables:

```text
second environment
shared model
10M to 20M model
```

Exit criterion:

```text
one checkpoint handles both environments
```

## Milestone 3: Transfer

Deliverables:

```text
third held-out environment
few-shot adaptation experiment
```

Exit criterion:

```text
pretraining materially reduces adaptation data versus scratch
```

This is the first major research milestone.

## Milestone 4: Natural language

Deliverables:

```text
small goal text encoder
paraphrase dataset
```

Exit criterion:

```text
natural-language goals approach structured-goal performance
```

## Milestone 5: Recovery and memory

Deliverables:

```text
perturbed trajectories
memory module
loop detection
recovery evaluation
```

Exit criterion:

```text
long-horizon episode success improves
```

## Milestone 6: Hierarchical control

Deliverables:

```text
subgoal head
offline subgoal annotations
```

Exit criterion:

```text
long tasks improve relative to flat policy
```

## Milestone 7: World model

Deliverables:

```text
latent dynamics model
counterfactual dataset
short lookahead
```

Exit criterion:

```text
lookahead improves action choice in ambiguous states
```

## Milestone 8: Web specialization

Deliverables:

```text
DOM adapter
pointer head
browser primitives
browser trajectory dataset
```

Exit criterion:

```text
general core fine-tunes faster than browser model trained from scratch
```

---

# 66. Immediate implementation tasks

The agent system should begin with these tasks only:

```text
Task A
Create repository structure.

Task B
Write schema definitions with validation.

Task C
Implement EnvironmentAdapter interface.

Task D
Implement TinyWorld procedural environment.

Task E
Implement deterministic oracle.

Task F
Generate 10k episodes.

Task G
Implement dataset loader.

Task H
Implement 5M MLX policy model.

Task I
Overfit on 100 episodes.

Task J
Train on full TinyWorld dataset.

Task K
Evaluate on unseen procedural layouts.

Task L
Produce first benchmark report.
```

Do not implement the world model before Task L is complete.

---

# 67. First benchmark report

The first report must contain:

```text
model parameter count
training transitions
validation transitions
test transitions
episode success
action accuracy
steps to success
latency p50
latency p95
peak memory
training duration
random baseline
oracle baseline
```

Also include 20 manually inspected failures.

---

# 68. Stop conditions

Pause and rethink architecture if:

```text
training accuracy is high but unseen layout success is near random

adding environments consistently destroys previous competence

the model relies heavily on environment IDs

small changes in entity ordering cause large output changes

progress prediction is uncorrelated with real progress

the action candidate representation becomes the bottleneck
```

Do not solve these by immediately increasing model size.

---

# 69. Sources and projects to inspect

Primary inspiration and related systems:

```text
Laya
https://github.com/NandhaKishorM/laya

Laya MLX
https://github.com/mizorewww/laya-mlx

Octo
https://octo-models.github.io/

Gato
https://deepmind.google/discover/blog/a-generalist-agent/

RT-1
https://research.google/blog/rt-1-robotics-transformer-for-real-world-control-at-scale/

BabyAI / MiniGrid
https://github.com/mila-iqia/babyai

ALFWorld
https://github.com/alfworld/alfworld

Mind2Web
https://github.com/OSU-NLP-Group/Mind2Web

WebLINX
https://github.com/mcgill-nlp/weblinx
```

Important note:

The project is not claiming that generalist policies, hierarchical RL, world models, pointer networks, latent actions, or transformer control are individually novel.

The research opportunity is in testing whether a small, non-autoregressive, domain-agnostic control architecture can combine these ideas into a practical replacement for LLM-heavy execution loops.

---

# 70. Final project statement

The target is not another chatbot.

The target is a small control model.

It receives:

```text
what I want
+
where I am
+
what exists
+
what I can do
+
what happened before
```

and returns:

```text
what to do next
+
how confident it is
+
whether progress is being made
+
whether the task is complete
```

The model should be:

```text
small
fast
local
non-autoregressive
trainable on consumer hardware
adaptable
domain-agnostic at its core
```

The long-term objective is:

> Build a reusable System 1 for agents: LLMs discover, explain and teach; the control model executes.
