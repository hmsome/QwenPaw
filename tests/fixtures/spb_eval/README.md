# SPB Evaluation Task Set

## Data Source

This evaluation set is adapted from **PersonaExt** (Zhu et al., PAED, ACL 2023).

- **Original dataset**: PersonaExt — based on PersonaChat annotations
  - 1,896 triplets / 6,357 utterance-triplet pairs
  - **License**: MIT
- **Upstream data**: PersonaChat (Zhang et al., ACL 2018)

## Adaptation Process

1. Download PersonaExt raw data → `raw/` (gitignored, rebuilt on demand)
2. Apply `mapping.json` to convert PersonaExt predicates → SPB schema fields
3. Score and rank by selection criteria (see `selection_criteria.md`)
4. Select top 20 personas with maximum dimension/field diversity
5. Manual review and adjustments logged in `build_log.md`

## Selection Criteria (Priority Order)

1. **Field density**: ≥3 triplets mappable to our schema
2. **Dimension diversity**: demographics / background / interests all well-covered
3. **List fields**: ≥5 samples with list-type fields (topics/hobbies/expertise_areas)
4. **Lexical heterogeneity**: ≥4 samples where dialogue wording ≠ ground-truth wording
5. **Natural turn distribution**: 5-turn dialogue slices with spread-out info

## Target Field Coverage

| Dimension.Field         | Target Coverage |
|------------------------|----------------|
| demographics.name      | 18/20          |
| background.occupation  | 16/20          |
| background.expertise_areas | 8/20      |
| interests.topics       | 18/20          |
| interests.hobbies      | 14/20          |
| communication.*        | N/A (ablation) |
| preferences.*          | 6/20           |

## Persona JSON Format

Each file `pe_XXX.json` contains:

```json
{
    "source": "PersonaExt:<original_id>",
    "ground_truth": {
        "demographics": {"name": "...", ...},
        "background": {...},
        ...
    },
    "dialogue_turns": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}
```

## Citation

```
@inproceedings{zhu2023paed,
    title={PAED: Zero-Shot Persona Attribute Extraction in Dialogues},
    author={Zhu, et al.},
    booktitle={ACL},
    year={2023}
}
@inproceedings{zhang2018personalizing,
    title={Personalizing Dialogue Agents: I have a dog, do you have pets too?},
    author={Zhang, et al.},
    booktitle={ACL},
    year={2018}
}
```
