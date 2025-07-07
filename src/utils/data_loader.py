import json
import logging
from typing import List, Union
from langchain.docstore.document import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_course_data(path: str = "src/data/courses.json") -> List[Document]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            course_data = json.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"Failed to load course data: File not found at {path}")
    except json.JSONDecodeError:
        raise RuntimeError(f"Failed to load course data: Invalid JSON format in {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load course data: {e}")

    if not isinstance(course_data, list):
        raise RuntimeError("Failed to load course data: Course data is not in the expected list format.")

    docs: List[Document] = []
    
    def normalize_to_list(value: Union[str, List[str]]) -> List[str]:
        if isinstance(value, list):
            return value
        elif isinstance(value, str):
            return [s.strip() for s in value.split(",")] if "," in value else [value.strip()]
        else:
            return []

    for item in course_data:
        if not isinstance(item, dict):
            continue

        subjects = normalize_to_list(item.get("subjects", []))
        grades = normalize_to_list(item.get("grades", []))

        content = (
            f"Title: {item.get('title', 'N/A')}\n"
            f"Description: {item.get('description', 'N/A').strip()}\n"
            f"Subjects: {', '.join(subjects)}\n"
            f"Grade: {', '.join(grades)}\n"
            f"isDualCredit: {item.get('isDualCredit', False)}\n"
            f"isCreditRecovery: {item.get('isCreditRecovery', False)}\n"
            f"HigherEdCredits: {item.get('higherEdCredits', 0)}"
        )

        metadata = {
            "courseId": item.get("courseId", "N/A"),
            "isFlex": item.get('isFlex', "N/A")
        }

        docs.append(Document(page_content=content, metadata=metadata))

    logger.info(f"Loaded {len(docs)} course documents.")
    return docs
