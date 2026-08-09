from typing import *
from numpy import ndarray

import math

import numpy as np
from shapely.geometry import Polygon, Point
from nltk.corpus import cmudict


def compute_loc_rel(corners1: ndarray, corners2: ndarray, name1: str, name2: str) -> Optional[str]:
    assert corners1.shape == corners2.shape == (8, 3), "Shape of corners should be (8, 3)."

    center1 = corners1.mean(axis=0)
    center2 = corners2.mean(axis=0)

    d = center1 - center2
    theta = math.atan2(d[2], d[0])  # range -pi to pi
    distance = (d[2]**2 + d[0]**2)**0.5  # center distance on the ground

    box1 = corners1[[0, 1, 4, 5], :][:, [0, 2]]  # 4 corners of the bottom face (0&5, 1&4 are opposite corners)
    box2 = corners2[[0, 1, 4, 5], :][:, [0, 2]]

    # Note that bounding boxes might not be axis-aligned
    polygon1, polygon2 = Polygon(box1[[0, 1, 3, 2], :]), Polygon(box2[[0, 1, 3, 2], :])  # change the order to be convex
    point1, point2 = Point(center1[[0, 2]]), Point(center2[[0, 2]])

    # Initialize the relationship
    p = None

    # Horizontal relationship: "left"/"right"/"front"/"behind"
    if theta >= 3 * math.pi / 4 or theta < -3 * math.pi / 4:
        p = "left of"
    elif -3 * math.pi / 4 <= theta < -math.pi / 4:
        p = "behind"
    elif -math.pi / 4 <= theta < math.pi / 4:
        p = "right of"
    elif math.pi / 4 <= theta < 3 * math.pi / 4:
        p = "in front of"

    # Vertical relationship: "above"/"below"
    if point1.within(polygon2) or point2.within(polygon1):
        delta1 = center1[1] - center2[1]
        delta2 = (
            corners1[:, 1].max() - corners1[:, 1].min() +
            corners2[:, 1].max() - corners2[:, 1].min()
        ) / 2.
        if (delta1 - delta2) >= 0. or "lamp" in name1:
            # Indicate that:
            # (1) delta1 > 0. (because always delta2 > 0.): `center1` is above `center2`
            # (2) delta1 >= delta2: `corners1` and `corners2` not intersect vertically
            # ==> `corners1` is completely above `corners2`
            # Or the subject is a lamp, which is always above other objects
            p = "above"
            return p
        if (-delta1 - delta2) >= 0. or "lamp" in name2:
            # ==> `corners1` is completely below `corners2`
            # Or the object is a lamp, which is always above other objects
            p = "below"
            return p

    if distance > 3.:
        return None  # too far away
    else:
        if distance < 1.:
            p = "closely " + p
        return p

def starts_with_vowel_sound(word, pronunciations=cmudict.dict()):
    for syllables in pronunciations.get(word, []):
        return syllables[0][-1].isdigit()

def get_article(word):
    word = word.split(" ")[0]
    article = "an" if starts_with_vowel_sound(word) else "a"
    return article

def reverse_rel(rel: str) -> str:
    return {
        "above": "below",
        "below": "above",
        "in front of": "behind",
        "behind": "in front of",
        "left of": "right of",
        "right of": "left of",
        "closely in front of": "closely behind",
        "closely behind": "closely in front of",
        "closely left of": "closely right of",
        "closely right of": "closely left of"
    }[rel]

def rotate_rel(rel: str, r: float) -> str:
    assert r in [0.0, np.pi * 0.5, np.pi, np.pi * 1.5]

    if rel in ["above", "below"]:
        return rel

    if r == 0.0:
        return rel
    elif r == np.pi * 0.5:
        return ("closely " if "closely " in rel else "") + \
            {
                "in front of": "right of",
                "behind": "left of",
                "left of": "in front of",
                "right of": "behind"
            }[rel.replace("closely ", "")]
    elif r == np.pi:
        return ("closely " if "closely " in rel else "") + \
            {
                "in front of": "behind",
                "behind": "in front of",
                "left of": "right of",
                "right of": "left of"
            }[rel.replace("closely ", "")]
    elif r == np.pi * 1.5:
        return ("closely " if "closely " in rel else "") + \
            {
                "in front of": "left of",
                "behind": "right of",
                "left of": "behind",
                "right of": "in front of"
            }[rel.replace("closely ", "")]

################################################################
class TextPreprocessorOriginal():
    def __init__(self,
                 object_types: List[str], 
                 predicate_types: List[str],
                 seed: Optional[int]=None):
        self.object_types = object_types
        self.predicate_types = predicate_types
        if seed is not None:
            np.random.seed(seed)

    def fill_templates(
        self,
        desc: Dict[str, List],
        object_descs: Optional[List[str]]=None,
        return_obj_ids=False,
        return_descs_no_dup=False,
        seed: Optional[int]=None,
        return_triplets=True,
    ) -> Tuple[str, Dict[int, int], List[Tuple[int, int, int]], List[Tuple[str, str]]]:
        if seed is not None:
            np.random.seed(seed)
        obj_class_ids = desc["obj_class_ids"]  # map from object index to class id

        # Describe the relations between the main objects and others
        selected_relation_indices = np.random.choice(
            len(desc["obj_relations"]),
            min(np.random.choice([1, 2]), len(desc["obj_relations"])),  # select 1 or 2 relations
            replace=False
        )
        selected_relations = [desc["obj_relations"][idx] for idx in selected_relation_indices]
        selected_relations = [
            (int(obj_class_ids[s]), int(p), int(obj_class_ids[o]))
            for s, p, o in selected_relations
        ]  # e.g., [(4, 2, 18), ...]; 4, 18 are class ids; 2 is predicate id
        selected_descs = []
        selected_sentences = []
        selected_object_ids, selected_relations_ids = [], []  # e.g., [0, ...]; 0 is object id
        for idx in selected_relation_indices:
            s, p, o = desc["obj_relations"][idx]
            s, p, o = int(s), int(p), int(o)
            if object_descs is None:
                s_name = self.object_types[obj_class_ids[s]].replace("_", " ")
                o_name = self.object_types[obj_class_ids[o]].replace("_", " ")
                p_str = self.predicate_types[p]
                if np.random.rand() > 0.5:
                    subject = f"{get_article(s_name).replace('a', 'A')} {s_name}"
                    predicate = f" is {p_str} "
                    object = f"{get_article(o_name)} {o_name}."
                else:  # 50% of the time to reverse the order
                    subject = f"{get_article(o_name).replace('a', 'A')} {o_name}"
                    predicate = f" is {reverse_rel(p_str)} "
                    object = f"{get_article(s_name)} {s_name}."
            else:
                if np.random.rand() < 0.75:
                    s_name = object_descs[s]
                else:  # 25% of the time to use the object type as the description
                    s_name = self.object_types[obj_class_ids[s]].replace("_", " ")
                    s_name = f"{get_article(s_name)} {s_name}"  # "a" or "an" is added
                if np.random.rand() < 0.75:
                    o_name = object_descs[o]
                else:
                    o_name = self.object_types[obj_class_ids[o]].replace("_", " ")
                    o_name = f"{get_article(o_name)} {o_name}"

                p_str = self.predicate_types[p]
                rev_p_str = reverse_rel(p_str)

                if p_str in ["left of", "right of"]:
                    if np.random.rand() < 0.5:
                        p_str = "to the " + p_str
                        rev_p_str = "to the " + rev_p_str
                elif p_str in ["closely left of", "closely right of"]:
                    if np.random.rand() < 0.25:
                        p_str = "closely to the " + p_str.split(" ")[-2] + " of"
                        rev_p_str = "closely to the " + rev_p_str.split(" ")[-2] + " of"
                    elif np.random.rand() < 0.5:
                        p_str = "to the close " + p_str.split(" ")[-2] + " of"
                        rev_p_str = "to the close " + rev_p_str.split(" ")[-2] + " of"
                    elif np.random.rand() < 0.75:
                        p_str = "to the near " + p_str.split(" ")[-2] + " of"
                        rev_p_str = "to the near " + rev_p_str.split(" ")[-2] + " of"

                if np.random.rand() < 0.5:
                    verbs = ["Place", "Put", "Position", "Arrange", "Add", "Set up"]
                    if "lamp" in s_name:
                        verbs += ["Hang", "Install"]
                    verb = verbs[np.random.choice(len(verbs))]
                    subject = f"{verb} {s_name}"
                    predicate = f" {p_str} "
                    object = f"{o_name}."
                    selected_relations_ids.append((s,o))
                    selected_descs.append((s_name, o_name))
                    selected_object_ids.append(s)
                else:  # 50% of the time to reverse the order
                    verbs = ["Place", "Put", "Position", "Arrange", "Add", "Set up"]
                    if "lamp" in o_name:
                        verbs += ["Hang", "Install"]
                    verb = verbs[np.random.choice(len(verbs))]
                    subject = f"{verb} {o_name}"
                    predicate = f" {rev_p_str} "
                    object = f"{s_name}."
                    selected_relations_ids.append((o, s))
                    selected_descs.append((o_name, s_name))
                    selected_object_ids.append(o)
            selected_sentences.append(subject + predicate + object)

        text = ""
        conjunctions = [" Then, ", " Next, ", " Additionally, ", " Finnally, ", " And ", " "]
        for i, sentence in enumerate(selected_sentences):
            if i == 0:
                text += sentence
            else:
                conjunction = conjunctions[np.random.choice(len(conjunctions))]
                while conjunction == " Finnally, " and i != len(selected_sentences)-1:
                    # "Finally" should be used only in the last sentence
                    conjunction = conjunctions[np.random.choice(len(conjunctions))]
                if conjunction != " ":
                    sentence = sentence[0].lower() + sentence[1:]
                text += conjunction + sentence

        returns = [text, selected_relations, selected_descs]
        obj_set = set()
        selected_descs_no_dup_ = {}
        for i, (obj1, obj2) in enumerate(selected_relations_ids):
            if obj1 not in obj_set:
                obj_set.add(obj1)
                selected_descs_no_dup_[obj1] = selected_descs[i][0]
            else:
                if len(selected_descs_no_dup_[obj1]) < len(selected_descs[i][0]):
                    selected_descs_no_dup_[obj1] = selected_descs[i][0]
            if obj2 not in obj_set:
                obj_set.add(obj2)
                selected_descs_no_dup_[obj2] = selected_descs[i][1]
            else:
                if len(selected_descs_no_dup_[obj2]) < len(selected_descs[i][1]):
                    selected_descs_no_dup_[obj2] = selected_descs[i][1]

        selected_descs_no_dup = []
        for key in selected_descs_no_dup_.keys():
            selected_descs_no_dup.append((selected_descs_no_dup_[key], obj_class_ids[key]))
        
        if return_obj_ids:
            returns.append(selected_object_ids)
        if return_descs_no_dup:
            returns.append(selected_descs_no_dup)
        if return_triplets:
            returns.append(1)
        return returns


class TextPreprocessor():
    def __init__(self,
                 object_types: List[str], 
                 predicate_types: List[str],
                 max_num_rel: int,
                 seed: Optional[int]=None):
        self.object_types = object_types
        self.predicate_types = predicate_types
        self.max_num_rel = max_num_rel
        self.drop_desc_rate = 0.25
        
        if seed is not None:
            np.random.seed(seed)
             
    def fill_templates(
        self,
        desc: Dict[str, Any],
        object_descs: Optional[List[str]]=None,
        return_obj_ids=False,
        return_triplets=False,
        return_descs_no_dup=False,
        seed: Optional[int]=None,
        no_reverse=False
        ) -> Tuple[str, Dict[int, int], List[Tuple[int, int, int]], List[Tuple[str, str]]]:
        if seed is not None:
            np.random.seed(seed)
        
        obj_class_ids = desc["obj_class_ids"]  # map from object index to class id
        relations = desc["obj_relations"]  # list of (subject, predicate, object) indices

        # Remove symmetric duplicates (ignore predicate for comparison)
        unique_relations = set()
        filtered_relations = []
        for s, p, o in relations:
            pair = tuple(sorted([s, o]))  # Ensure order consistency for comparison
            if pair not in unique_relations:
                unique_relations.add(pair)
                filtered_relations.append((s, p, o))
        
        # Handle case when filtered_relations is empty
        assert len(filtered_relations) > 0, f"Empty filtered_relations in desc: {desc}"
                
        # Randomly select a subset of filtered_relations
        max_relations = min(self.max_num_rel, len(filtered_relations))
        assert max_relations >= 1, f"max_relations < 1: {max_relations}, filtered_relations: {filtered_relations}"
            
        num_to_select = np.random.randint(1, max_relations + 1)
        selected_relation_indices = np.random.choice(len(filtered_relations), num_to_select, replace=False)
        
        # Track previously appeared objects
        mentioned_objects = set()
        sorted_indices = []
        must_fix_idx = []
        
        # Select first relation randomly
        first_idx = selected_relation_indices[0]
        s, p, o = filtered_relations[first_idx]
        mentioned_objects.add(s)
        mentioned_objects.add(o)
        sorted_indices.append(first_idx)
        
        # Sort remaining relations to use previously mentioned objects as objects
        remaining_indices = selected_relation_indices[1:]
        for idx in remaining_indices:
            s, p, o = filtered_relations[idx]
            if s in mentioned_objects:
                # If s was mentioned before, reverse the relation to use o as subject
                filtered_relations[idx] = (o, self.predicate_types.index(reverse_rel(self.predicate_types[p])), s)
                mentioned_objects.add(o)
                must_fix_idx.append(idx)
            else:
                mentioned_objects.add(s)
                mentioned_objects.add(o)
            sorted_indices.append(idx)
                
        selected_sentences = []
        selected_object_ids = []
        selected_descs = []
        selected_descs_no_dup = []
        
        first_mention = set() # Track if each object appeared for the first time
        last_mentioned = set() # Track objects mentioned in the previous sentence
        mentioned_types = set()# Track types of all previously mentioned objects

        rel_objs = [filtered_relations[idx] for idx in sorted_indices]
        selected_relations = [(int(obj_class_ids[s]), int(p), int(obj_class_ids[o])) for s, p, o in rel_objs]
        
        for i, idx in enumerate(sorted_indices):
            s, p, o = filtered_relations[idx]
            s, p, o = int(s), int(p), int(o)
            
            # Generate names for each object
            if object_descs is not None:
                # Generate name for subject(s)
                if s in last_mentioned:
                    # Use 'the' for objects mentioned in the previous sentence
                    s_name = f"the {self.object_types[obj_class_ids[s]].replace('_', ' ')}"
                elif s not in first_mention and obj_class_ids[s] in mentioned_types:
                    # Use 'another' for previously mentioned objects of the same type
                    s_name = f"another {self.object_types[obj_class_ids[s]].replace('_', ' ')}"
                    selected_descs_no_dup.append((s_name, obj_class_ids[s]))
                else:
                    # Object appearing for the first time
                    if np.random.rand() > self.drop_desc_rate:
                        s_name = object_descs[s]
                    else:
                        s_name = self.object_types[obj_class_ids[s]].replace("_", " ")
                        s_name = f"{get_article(s_name)} {s_name}"

                    if s not in first_mention:
                        selected_descs_no_dup.append((s_name, obj_class_ids[s]))
                    first_mention.add(s)
                
                # Generate name for object(o)
                if o in last_mentioned:
                    # Use 'the' for objects mentioned in the previous sentence
                    o_name = f"the {self.object_types[obj_class_ids[o]].replace('_', ' ')}"
                elif o not in first_mention and obj_class_ids[o] in mentioned_types:
                    # Use 'another' for previously mentioned objects of the same type
                    o_name = f"another {self.object_types[obj_class_ids[o]].replace('_', ' ')}"
                    selected_descs_no_dup.append((o_name, obj_class_ids[o]))
                else:
                    # Object appearing for the first time
                    if np.random.rand() > self.drop_desc_rate:
                        o_name = object_descs[o]
                    else:
                        o_name = self.object_types[obj_class_ids[o]].replace("_", " ")
                        o_name = f"{get_article(o_name)} {o_name}"

                    if o not in first_mention:
                        selected_descs_no_dup.append((o_name, obj_class_ids[o]))
                    first_mention.add(o)
            else:
                s_name = self.object_types[obj_class_ids[s]].replace("_", " ")
                o_name = self.object_types[obj_class_ids[o]].replace("_", " ")
                s_name = f"{get_article(s_name)} {s_name}"
                o_name = f"{get_article(o_name)} {o_name}"
            
            p_str = self.predicate_types[p]
            rev_p_str = reverse_rel(p_str)

            if p_str in ["left of", "right of"]:
                if np.random.rand() < 0.5:
                    p_str = "to the " + p_str
                    rev_p_str = "to the " + rev_p_str
            elif p_str in ["closely left of", "closely right of"]:
                if np.random.rand() < 0.25:
                    p_str = "closely to the " + p_str.split(" ")[-2] + " of"
                    rev_p_str = "closely to the " + rev_p_str.split(" ")[-2] + " of"
                elif np.random.rand() < 0.5:
                    p_str = "to the close " + p_str.split(" ")[-2] + " of"
                    rev_p_str = "to the close " + rev_p_str.split(" ")[-2] + " of"
                elif np.random.rand() < 0.75:
                    p_str = "to the near " + p_str.split(" ")[-2] + " of"
                    rev_p_str = "to the near " + rev_p_str.split(" ")[-2] + " of"

            if not no_reverse:
                if idx in must_fix_idx or np.random.rand() < 0.5:
                    verbs = ["Place", "Put", "Position", "Arrange", "Add", "Set up"]
                    if "lamp" in s_name:
                        verbs += ["Hang", "Install"]
                    verb = verbs[np.random.choice(len(verbs))]
                    subject = f"{verb} {s_name}"
                    predicate = f" {p_str} "
                    object = f"{o_name}."
                    selected_descs.append((s_name, o_name))
                    selected_object_ids.append(s)
                else:  # 50% of the time to reverse the order
                    verbs = ["Place", "Put", "Position", "Arrange", "Add", "Set up"]
                    if "lamp" in o_name:
                        verbs += ["Hang", "Install"]
                    verb = verbs[np.random.choice(len(verbs))]
                    subject = f"{verb} {o_name}"
                    predicate = f" {rev_p_str} "
                    object = f"{s_name}."
                    selected_descs.append((o_name, s_name))
                    selected_object_ids.append(o)
            else:
                verbs = ["Place", "Put", "Position", "Arrange", "Add", "Set up"]
                if "lamp" in s_name:
                    verbs += ["Hang", "Install"]
                verb = verbs[np.random.choice(len(verbs))]
                subject = f"{verb} {s_name}"
                predicate = f" {p_str} "
                object = f"{o_name}."
                selected_descs.append((s_name, o_name))
                selected_object_ids.append(s)


            selected_sentences.append(subject + predicate + object)
            
            # Record objects mentioned in current sentence and their types
            last_mentioned = {s, o}
            mentioned_types.add(obj_class_ids[s])
            mentioned_types.add(obj_class_ids[o])

        text = ""
        conjunctions = [" Then, ", " Next, ", " Additionally, ", " Finally, ", " And ", " "]
        for i, sentence in enumerate(selected_sentences):
            if i == 0:
                text += sentence
            else:
                conjunction = conjunctions[np.random.choice(len(conjunctions))]
                while conjunction == " Finally, " and i != len(selected_sentences)-1:
                    # "Finally" should be used only in the last sentence
                    conjunction = conjunctions[np.random.choice(len(conjunctions))]
                if conjunction != " ":
                    sentence = sentence[0].lower() + sentence[1:]
                text += conjunction + sentence

        returns = [text, selected_relations, selected_descs] # return `selected_relations`, `selected_descs` for evaluation
        if return_obj_ids:
            returns.append(selected_object_ids)
        if return_descs_no_dup:
            returns.append(selected_descs_no_dup)
        if return_triplets:
            returns.append(rel_objs)
        return returns


def fill_templates_whole(
    desc: Dict[str, List],
    object_types: List[str], 
    predicate_types: List[str],
    object_descs: Optional[List[str]]=None,
    seed: Optional[int]=None,
    return_obj_ids=False
) -> Tuple[str, Dict[int, int], List[Tuple[int, int, int]], List[Tuple[str, str]]]:
    if object_descs is None:
        assert object_types is not None

    if seed is not None:
        np.random.seed(seed)

    obj_class_ids = desc["obj_class_ids"]  # map from object index to class id

    # Describe the relations between the main objects and others
    selected_relations = desc["obj_relations"]
    selected_relations = [
        (int(obj_class_ids[s]), int(p), int(obj_class_ids[o]))
        for s, p, o in selected_relations
    ]  # e.g., [(4, 2, 18), ...]; 4, 18 are class ids; 2 is predicate id
    selected_descs = []
    selected_sentences = []
    selected_object_ids = []  # e.g., [0, ...]; 0 is object id
    for idx in range(len(desc["obj_relations"])):
        s, p, o = desc["obj_relations"][idx]
        s, p, o = int(s), int(p), int(o)
        if object_descs is None:
            s_name = object_types[obj_class_ids[s]].replace("_", " ")
            o_name = object_types[obj_class_ids[o]].replace("_", " ")
            p_str = predicate_types[p]
            if np.random.rand() > 0.5:
                subject = f"{get_article(s_name).replace('a', 'A')} {s_name}"
                predicate = f" is {p_str} "
                object = f"{get_article(o_name)} {o_name}."
            else:  # 50% of the time to reverse the order
                subject = f"{get_article(o_name).replace('a', 'A')} {o_name}"
                predicate = f" is {reverse_rel(p_str)} "
                object = f"{get_article(s_name)} {s_name}."
        else:
            if np.random.rand() < 0.75:
                s_name = object_descs[s]
            else:  # 25% of the time to use the object type as the description
                s_name = object_types[obj_class_ids[s]].replace("_", " ")
                s_name = f"{get_article(s_name)} {s_name}"  # "a" or "an" is added
            if np.random.rand() < 0.75:
                o_name = object_descs[o]
            else:
                o_name = object_types[obj_class_ids[o]].replace("_", " ")
                o_name = f"{get_article(o_name)} {o_name}"

            p_str = predicate_types[p]
            rev_p_str = reverse_rel(p_str)

            if p_str in ["left of", "right of"]:
                if np.random.rand() < 0.5:
                    p_str = "to the " + p_str
                    rev_p_str = "to the " + rev_p_str
            elif p_str in ["closely left of", "closely right of"]:
                if np.random.rand() < 0.25:
                    p_str = "closely to the " + p_str.split(" ")[-2] + " of"
                    rev_p_str = "closely to the " + rev_p_str.split(" ")[-2] + " of"
                elif np.random.rand() < 0.5:
                    p_str = "to the close " + p_str.split(" ")[-2] + " of"
                    rev_p_str = "to the close " + rev_p_str.split(" ")[-2] + " of"
                elif np.random.rand() < 0.75:
                    p_str = "to the near " + p_str.split(" ")[-2] + " of"
                    rev_p_str = "to the near " + rev_p_str.split(" ")[-2] + " of"

            if np.random.rand() < 0.5:
                verbs = ["Place", "Put", "Position", "Arrange", "Add", "Set up"]
                if "lamp" in s_name:
                    verbs += ["Hang", "Install"]
                verb = verbs[np.random.choice(len(verbs))]
                subject = f"{verb} {s_name}"
                predicate = f" {p_str} "
                object = f"{o_name}."
                selected_descs.append((s_name, o_name))
                selected_object_ids.append(s)
            else:  # 50% of the time to reverse the order
                verbs = ["Place", "Put", "Position", "Arrange", "Add", "Set up"]
                if "lamp" in o_name:
                    verbs += ["Hang", "Install"]
                verb = verbs[np.random.choice(len(verbs))]
                subject = f"{verb} {o_name}"
                predicate = f" {rev_p_str} "
                object = f"{s_name}."
                selected_descs.append((o_name, s_name))
                selected_object_ids.append(o)
        selected_sentences.append(subject + predicate + object)

    text = ""
    conjunctions = [" Then, ", " Next, ", " Additionally, ", " Finally, ", " And ", " "]
    for i, sentence in enumerate(selected_sentences):
        if i == 0:
            text += sentence
        else:
            conjunction = conjunctions[np.random.choice(len(conjunctions))]
            while conjunction == " Finally, " and i != len(selected_sentences)-1:
                # "Finally" should be used only in the last sentence
                conjunction = conjunctions[np.random.choice(len(conjunctions))]
            if conjunction != " ":
                sentence = sentence[0].lower() + sentence[1:]
            text += conjunction + sentence

    if return_obj_ids:
        return text, selected_relations, selected_descs, selected_object_ids
    else:
        return text, selected_relations, selected_descs  # return `selected_relations`, `selected_descs` for evaluation

def extract_random_triplets(
    desc: Dict[str, List],
    object_types: List[str], 
    predicate_types: List[str],
    object_descs: Optional[List[str]]=None,
    seed: Optional[int] = None
    ) -> str:
    if object_descs is None:
        assert object_types is not None
    
    if seed is not None:
        np.random.seed(seed)

    obj_class_ids = desc["obj_class_ids"]  # map from object index to class id
    relations = desc["obj_relations"]  # list of (subject, predicate, object) indices

    # Remove symmetric duplicates (ignore predicate for comparison)
    unique_relations = set()
    filtered_relations = []
    for s, p, o in relations:
        pair = tuple(sorted([s, o]))  # Ensure order consistency for comparison
        if pair not in unique_relations:
            unique_relations.add(pair)
            filtered_relations.append((s, p, o))
            
    # Randomly select a subset of filtered_relations
    if len(filtered_relations)>1:
        num_to_select = np.random.randint(1, min(5,len(filtered_relations)))
        selected_triplets_id = np.random.choice(len(filtered_relations), num_to_select, replace=False)
        filtered_relations = [filtered_relations[idx] for idx in selected_triplets_id]
        
    # Extract all unique triplets (avoid symmetric duplicates)
    selected_descs = []
    triplets = set()
    for s, p, o in filtered_relations:
        s, p, o = int(s), int(p), int(o)

        # Handle object descriptions
        if np.random.rand() < 0.75:
            s_name = object_descs[s]
        else:  # 25% probability to use only object names without description
            s_name = object_types[obj_class_ids[s]].replace("_", " ")
            s_name = f"{get_article(s_name)} {s_name}"  # "a" or "an" is added
        if np.random.rand() < 0.75:
            o_name = object_descs[o]
        else:
            o_name = object_types[obj_class_ids[o]].replace("_", " ")
            o_name = f"{get_article(o_name)} {o_name}"

        p_str = predicate_types[p]
        rev_p_str = reverse_rel(p_str)
        # 50% probability to reverse
        if np.random.rand() < 0.5:
            triplets.add((s_name, p_str, o_name))
            selected_descs.append((s_name, o_name))
        else:
            triplets.add((o_name, rev_p_str, s_name))
            selected_descs.append((o_name, s_name))

    # triplets = list(triplets)
    
    # Format the triplets as a string
    triplet_strings = [f"({subj}, {rel}, {obj})" for subj, rel, obj in triplets]

    return triplet_strings, filtered_relations, selected_descs

# def inst_and_triplets(
#     desc: Dict[str, List],
#     object_types: List[str], 
#     predicate_types: List[str],
#     object_descs: Optional[List[str]]=None,
#     seed: Optional[int]=None,
#     return_obj_ids=False
# )


def extract_triplet2( #return (rel_text), (obj_Text)
    desc: Dict[str, List],
    object_types: List[str], 
    predicate_types: List[str],
    object_descs: Optional[List[str]]=None,
    seed: Optional[int] = None
    ) -> str:
    
    if seed is not None:
        np.random.seed(seed)

    obj_class_ids = desc["obj_class_ids"]  # map from object index to class id
    relations = desc["obj_relations"]  # list of (subject, predicate, object) indices

    # Remove symmetric duplicates (ignore predicate for comparison)
    unique_relations = set()
    filtered_relations = []
    for s, p, o in relations:
        pair = tuple(sorted([s, o]))  # Ensure order consistency for comparison
        if pair not in unique_relations:
            unique_relations.add(pair)
            filtered_relations.append((s, p, o))
            
    # Randomly select a subset of filtered_relations
    if len(filtered_relations)>1:
        num_to_select = np.random.randint(1, min(5,len(filtered_relations)))
        selected_triplets_id = np.random.choice(len(filtered_relations), num_to_select, replace=False)
        filtered_relations = [filtered_relations[idx] for idx in selected_triplets_id]
        
    # Extract all unique triplets (avoid symmetric duplicates)
    selected_descs = []
    triplets = []
    for s, p, o in filtered_relations:
        s, p, o = int(s), int(p), int(o)
        # Handle object descriptions
        s_obj = object_types[obj_class_ids[s]].replace("_", " ")
        if np.random.rand() < 0.75:
            s_desc = object_descs[s]
        else:  # 25% probability to use only object names without description
            s_desc = s_obj
        o_obj = object_types[obj_class_ids[o]].replace("_", " ")
        if np.random.rand() < 0.75:
            o_desc = object_descs[o]
        else:  # 25% probability to use only object names without description
            o_desc = o_obj

        p_str = predicate_types[p]
        rev_p_str = reverse_rel(p_str)
        # 50% probability to reverse
        if np.random.rand() < 0.5:
            triplets.append(f"{s_obj} {p_str} {o_obj}")
            selected_descs.append([s_desc, o_desc])
        else:
            triplets.append(f"{o_obj} {rev_p_str} {s_obj}")
            selected_descs.append([o_desc, s_desc])

    return triplets, filtered_relations, selected_descs



