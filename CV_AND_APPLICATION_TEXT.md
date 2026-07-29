# Interview talk track

## Problem

Football analysts need reliable tools that turn event data into interpretable evidence without breaking existing video and coaching workflows.

## Product choices

I built one shared analytics layer and exposed it through an interactive application, an API and a report. This avoids re-implementing definitions in several places.

## Engineering choices

I used transparent baseline models, match-aware evaluation, typed API inputs, modular code, automated tests, Docker and CI. The project is deliberately provider-neutral.

## Research choices

The xG model is evaluated for ranking, probability accuracy and calibration. The xT implementation exposes assumptions and convergence. Every metric includes a documented definition and limitation.

## User focus

The dashboard is organised by real workflows: match review, shot quality, possession threat, passing structure, recruitment and model governance. Each view is designed to support a conversation with analysts and coaches rather than to replace judgement.

## Honest limitations

The bundled data is synthetic. A professional club could use the architecture as a starting point, but operational use requires licensed data, domain validation, monitoring, security and integration into the club's workflow.
