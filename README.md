# natural-archive-pipeline

Map restoration and print-file pipeline for The Natural Archive (v2.6), run on GitHub Actions.
Non-generative: crop, fold lighting flatten, neatline dewarp, paper extension from the scan's own margins.

Triggered by `workflow_dispatch` (`job_id`, `request` JSON). Results land on the `results` branch as `results/<job_id>.json`.
