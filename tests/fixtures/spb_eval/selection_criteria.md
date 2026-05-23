# Selection Criteria for SPB Evaluation Personas

## Scoring Rules

Each PersonaExt persona is scored (0–10) on the following criteria:

### 1. Field Density (weight: 3)
- **3 points**: ≥5 triplets mappable to SPB schema
- **2 points**: 3–4 mappable triplets
- **1 point**: 1–2 mappable triplets
- **0 points**: no mappable triplets

### 2. Dimension Diversity (weight: 3)
- **3 points**: covers 3+ distinct SPB dimensions
- **2 points**: covers 2 dimensions
- **1 point**: covers 1 dimension only

### 3. List Field Coverage (weight: 2)
- **2 points**: includes ≥1 list-type field (topics/hobbies/expertise_areas)
- **1 point**: list field present but with single item
- **0 points**: no list fields

### 4. Lexical Heterogeneity (weight: 1)
- **1 point**: dialogue wording differs from ground-truth labeling
- **0 points**: wording is identical

### 5. Turn Distribution (weight: 1)
- **1 point**: profile info spread across ≥3 turns
- **0 points**: all info in first turn

## Selection Process

1. Score all PersonaExt personas (1886 triplets)
2. Sort by total score descending
3. Enforce minimum coverage targets per field (see README.md target table)
4. Greedily select top 20 ensuring field coverage targets are met
5. Manual review of each selected persona
