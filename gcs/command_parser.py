import re
import logging

logger = logging.getLogger(__name__)

SYNONYM_MAP = {
    "bottle": ["bottle", "bottles", "water bottle", "flask", "canteen", "container"],
    "book": ["book", "books", "notebook", "notebooks", "textbook", "novel", "magazine", "pad", "diary", "journal"],
    "pen": ["pen", "pens", "pencil", "pencils", "marker", "stylus", "ballpoint"],
    "person": ["person", "people", "human", "humans", "man", "woman", "guy", "pedestrian", "individual", "tshirt", "t-shirt", "shirt"],
    "car": ["car", "cars", "vehicle", "vehicles", "automobile", "automobiles"],
    "truck": ["truck", "trucks", "lorry", "pickup"],
    "bus": ["bus", "busses", "buses"],
    "boat": ["boat", "boats", "ship", "vessel", "watercraft"],
    "cell phone": ["cell phone", "phone", "phones", "mobile", "cellphone", "smartphone"],
    "backpack": ["backpack", "bag", "bags", "backpacks", "rucksack", "handbag"],
    "chair": ["chair", "chairs", "seat", "stool"],
    "dog": ["dog", "dogs", "puppy", "canine"],
    "cat": ["cat", "cats", "kitten", "feline"],
    "bicycle": ["bicycle", "bicycles", "bike", "bikes", "cycle"],
    "laptop": ["laptop", "laptops", "computer", "macbook"]
}

COLOR_WORDS = ["red", "green", "blue", "yellow", "orange", "purple", "black", "white", "gray", "brown", "cyan", "dark"]

class CommandParser:
    def __init__(self, wake_word="jarvis"):
        self.wake_word = wake_word.lower()

    def parse_command(self, text_input):
        if not text_input or not isinstance(text_input, str):
            return {"action": "UNKNOWN", "params": {}, "raw": str(text_input)}

        raw_text = text_input.strip()
        clean_text = raw_text.lower()

        # Strip Vosk [unk] tokens (grammar-mode artifact)
        clean_text = re.sub(r'\[unk\]\s*', '', clean_text).strip()

        is_voice = self.wake_word in clean_text
        if is_voice:
            clean_text = clean_text.replace(self.wake_word, "").strip()

        clean_text = re.sub(r'^[^\w\s]+', '', clean_text).strip()

        # Strip common voice filler / politeness words at the start
        clean_text = re.sub(
            r'^(ok|okay|hey|drone|jarvis|computer|now|just|only|please|go|go and|'
            r'start|begin|can you|could you|i want you to|i want to|'
            r'let\'s|lets|try to|try|make it|make sure to)\s+',
            '', clean_text
        ).strip()

        # Reject trivially short / stop-word-only phrases
        _STOP_WORDS = {"the", "a", "an", "that", "this", "it", "ok", "okay",
                       "yes", "no", "at", "of", "on", "in", "up", "go", "and"}
        if not clean_text or clean_text in _STOP_WORDS:
            return {"action": "UNKNOWN", "params": {}, "raw": raw_text,
                    "source": "voice" if is_voice else "text"}

        if any(clean_text == kw or clean_text.startswith(kw) for kw in ["arm", "arm motors", "start motors"]):
            return {"action": "ARM", "params": {}, "raw": raw_text, "source": "voice" if is_voice else "text"}
        if any(clean_text == kw or clean_text.startswith(kw) for kw in ["disarm", "stop motors", "disarm motors"]):
            return {"action": "DISARM", "params": {}, "raw": raw_text, "source": "voice" if is_voice else "text"}

        takeoff_match = re.search(r"(?:takeoff|fly up|ascend)(?:\s+(?:to\s+)?(\d+(?:\.\d+)?)\s*m(?:eters)?)?", clean_text)
        if takeoff_match or clean_text.startswith("takeoff"):
            alt = float(takeoff_match.group(1)) if takeoff_match and takeoff_match.group(1) else 10.0
            return {"action": "TAKEOFF", "params": {"altitude": alt}, "raw": raw_text, "source": "voice" if is_voice else "text"}

        if any(kw in clean_text for kw in ["land", "touch down"]):
            return {"action": "LAND", "params": {}, "raw": raw_text, "source": "voice" if is_voice else "text"}
        if any(kw in clean_text for kw in ["rtl", "return home", "return to launch", "go home"]):
            return {"action": "RTL", "params": {}, "raw": raw_text, "source": "voice" if is_voice else "text"}
        if any(kw in clean_text for kw in ["hold", "loiter", "pause"]):
            return {"action": "HOLD", "params": {}, "raw": raw_text, "source": "voice" if is_voice else "text"}
        if any(kw in clean_text for kw in ["scan_geo", "scan geo", "scan geography", "scan environment", "scan terrain", "scangeo"]):
            return {"action": "SCAN_GEO", "params": {"terrain_type": "all"}, "raw": raw_text, "source": "voice" if is_voice else "text"}

        # Use regex word-boundary search (not startswith) so filler words
        # before the action keyword still match correctly.
        _TRACK_RE  = re.compile(r'\b(track|follow|lock on|lock|pursue|chase|keep on)\b')
        _SEARCH_RE = re.compile(r'\b(search|find|scan for|look for|locate|detect|search for|identify|spot)\b')

        if _TRACK_RE.search(clean_text):
            requested_action = "TRACK"
            query_body = _TRACK_RE.sub('', clean_text, count=1)
            query_body = re.sub(r'^\s*(id\s*)?', '', query_body).strip()
            query_body = re.sub(r'^(a|the|an|that|this|one)\s+', '', query_body).strip()
        else:
            requested_action = "SEARCH"
            query_body = _SEARCH_RE.sub('', clean_text, count=1)
            query_body = re.sub(r'^\s*(for\s+)?(a|the|an|that|specific|any|one)\s+', '', query_body).strip()

        if requested_action == "TRACK" and query_body.isdigit():
            return {"action": "TRACK", "params": {"target_id": query_body}, "raw": raw_text, "source": "voice" if is_voice else "text"}

        ocr_text_query = None
        text_clause_match = re.search(r"(?:with|having|written|text|named|saying|printed)\s+(?:text|word|name)?\s*[\'\"]?([a-zA-Z0-9]+)[\'\"]?", query_body)
        if text_clause_match:
            ocr_text_query = text_clause_match.group(1).strip()
            if ocr_text_query in ["written", "printed", "with", "on", "tshirt", "shirt", "text"]:
                words = [w for w in raw_text.split() if w.isupper() or (w.istitle() and w.lower() not in ["search", "track", "person", "with", "written", "on", "tshirt", "shirt"])]
                if words:
                    ocr_text_query = words[0]

        detected_color = None
        for color in COLOR_WORDS:
            if color in query_body:
                detected_color = color
                break

        target_str = query_body
        if detected_color:
            target_str = re.sub(rf'\b{detected_color}\b', '', target_str).strip()

        normalized_target = target_str
        for base_class, synonyms in SYNONYM_MAP.items():
            if any(syn in target_str for syn in synonyms) or base_class in target_str:
                normalized_target = base_class
                break

        # Reject results with no useful target for SEARCH/TRACK
        # (catches Vosk mishears like "at the person" where target would be "at")
        _STOP_WORDS = {"the", "a", "an", "that", "this", "it", "at", "of",
                       "on", "in", "up", "and", "lock", "[unk]"}
        if normalized_target in _STOP_WORDS or not normalized_target:
            return {"action": "UNKNOWN", "params": {}, "raw": raw_text,
                    "source": "voice" if is_voice else "text"}

        return {
            "action": requested_action,
            "params": {
                "query": query_body,
                "target": normalized_target,
                "color": detected_color,
                "text_query": ocr_text_query
            },
            "raw": raw_text,
            "source": "voice" if is_voice else "text"
        }
