import argparse
from datasets import load_dataset
import json
import os

CONVERSION_DICT = {
    "Facility": "location-facility",
    "OtherLOC": "location-other",
    "HumanSettlement": "location-human_settlement",
    "Station": "location-station",
    "VisualWork": "creative-visual",
    "MusicalWork": "creative-music",
    "WrittenWork": "creative-written",
    "ArtWork": "creative-art",
    "Software": "creative-software",
    "MusicalGRP": "group-musical",
    "PublicCorp": "group-public_corp",
    "PrivateCorp": "group-private_corp",
    "AerospaceManufacturer": "group-aerospace_manufacturer",
    "SportsGRP": "group-sports",
    "CarManufacturer": "group-car_manufacturer",
    "ORG": "group-organization",
    "Scientist": "person-scientist",
    "Artist": "person-artist",
    "Athlete": "person-athlete",
    "Politician": "person-politician",
    "Cleric": "person-cleric",
    "SportsManager": "person-sports_manager",
    "OtherPER": "person-other",
    "Clothing": "product-clothing",
    "Vehicle": "product-vehicle",
    "Food": "product-food",
    "Drink": "product-drink",
    "OtherPROD": "product-other",
    "Medication/Vaccine": "medical-medication",
    "MedicalProcedure": "medical-procedure",
    "AnatomicalStructure": "medical-anatomy",
    "Symptom": "medical-symptom",
    "Disease": "medical-disease"
}

# POS Tagger
def pos_tag_tokens(tokens):
    pos_tags = ["NOUN" for token in tokens]

    return pos_tags


def convert_tokens(sample):
    """
    Creates a dictionary that maps tokens to their corresponding POS tags.

    Args:
        sample : A data sample containing tokens, NER tags and id

    Returns:
        converted_dict: a dictionary that maps the MultiCoNER sample to a PIQN sample.
    """

    converted_dict = {
        "tokens": sample['tokens'],
        "entities": extract_entities(sample['ner_tags']),
        "org_id": sample["id"],
        "relations": {},
        "pos": pos_tag_tokens(sample['tokens']),
        "ltokens": [],
        "rtokens": [],
    }

    return converted_dict


def extract_entities(ner_tags):
    entities = []
    entity_dict = {}
    entity = ""
    start = -1
    for idx, tag in enumerate(ner_tags):
        if tag == "O":
            if entity == "":
                continue
            elif start!=-1:
                entity_dict["type"] = entity
                entity_dict["start"] = start
                entity_dict["end"] = idx
                entities.append(entity_dict)
                entity = ""
                start = -1
                entity_dict = {}
        elif tag.split("-")[0] == "B":
            entity = CONVERSION_DICT[tag.split("-")[1]]
            start = idx
        elif tag.split("-")[0] == "I" and entity != "":
            continue

    return entities

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="German (DE)")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--save-dir", type=str, default="./data/multiconer")

    args = parser.parse_args()
    language = args.language
    split = args.split
    save_dir = args.save_dir

    dataset = load_dataset("MultiCoNER/multiconer_v2", language)
    converted_json = [convert_tokens(example) for example in dataset[split]]

    with open(save_dir + os.path.sep + f"MultiCoNER_{language.split()[0]}_{split}.json", "w") as f:
        json.dump(converted_json, f)