# Notice

This repository is a **reference example**: it shows how to drive an Amazon SageMaker AI
**deploy + managed performance benchmark** workflow for an open-weight LLM through a coding
agent, using two portable **SKILL.md** contracts.

- Provided **as-is**, for educational / reference use. Review the code before running it in
  your own account.
- The scripts create **billable** AWS resources (GPU real-time endpoints). They bill while
  `InService` — tear them down when done with `scripts/teardown.py` (deletes the inference
  component → endpoint → endpoint config → model).
- You are responsible for **model licenses** (e.g. the open-weight model you deploy), **AWS
  costs**, **IAM permissions**, and **service quotas** in your account and region.
- Nothing in this repo is account-specific — region, account, execution role, and bucket are
  resolved from your environment at runtime (`scripts/config.py`). Quota does **not**
  guarantee capacity — keep a fallback instance in mind.

No warranty. Validate in a non-production account first.
