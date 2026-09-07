import ast
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

client = OpenAI()

MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5.6-luna"
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_SOURCE_DIR = BASE_DIR / "Data" / "Source"

CONTEXT_DIR = BASE_DIR / "context"

CONTEXT_PROMPT_FILE = (
    CONTEXT_DIR / "context_input_parser.txt"
)

ASSET_PROFILE_FILE = (
    CONTEXT_DIR / "asset_profiles.json"
)

QUESTION_FILE = (
    CONTEXT_DIR / "question_input_parser.txt"
)

TEST_RESULT_FILE = (
    CONTEXT_DIR / "test_results_input_parser.txt"
)


# =========================================================
# CONSTANTS
# =========================================================

ALLOWED_INTENTS = {
    "status",
    "summary",
    "trend",
    "risk",
    "opportunity",
    "outlook",
    "sentiment",
    "drivers",
    "impact",
    "comparison",
    "valuation",
    "fundamentals",
    "scenario"
}


# =========================================================
# HELPERS
# =========================================================

def normalize_text(value: str) -> str:

    value = value.strip().lower()

    value = value.replace(
        "_",
        " "
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value


# =========================================================
# DATA SOURCE KEYWORDS
# =========================================================

def get_available_keywords() -> list[str]:

    if not DATA_SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Data source folder bulunamadı: "
            f"{DATA_SOURCE_DIR}"
        )

    keywords = [
        folder.name
        for folder in DATA_SOURCE_DIR.iterdir()
        if folder.is_dir()
    ]

    return sorted(
        keywords
    )


# =========================================================
# ASSET PROFILES
# =========================================================

def load_asset_profiles() -> dict:

    if not ASSET_PROFILE_FILE.exists():
        raise FileNotFoundError(
            f"Asset profile file bulunamadı: "
            f"{ASSET_PROFILE_FILE}"
        )

    profiles = json.loads(
        ASSET_PROFILE_FILE.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        profiles,
        dict
    ):
        raise ValueError(
            "asset_profiles.json root object "
            "bir dictionary olmalı."
        )

    return profiles


# =========================================================
# CONTEXT PROMPT
# =========================================================

def load_context_prompt() -> str:

    if not CONTEXT_PROMPT_FILE.exists():
        raise FileNotFoundError(
            f"Context prompt bulunamadı: "
            f"{CONTEXT_PROMPT_FILE}"
        )

    return CONTEXT_PROMPT_FILE.read_text(
        encoding="utf-8"
    )


# =========================================================
# KEYWORD RESOLVER
# =========================================================

def resolve_keyword(
    subject: str | None,
    available_keywords: list[str],
    profiles: dict
) -> str | None:

    if subject is None:
        return None

    normalized_subject = normalize_text(
        subject
    )

    # -----------------------------------------------------
    # 1. Direct Data/Source folder match
    # -----------------------------------------------------

    for keyword in available_keywords:

        if (
            normalize_text(keyword)
            == normalized_subject
        ):
            return keyword

    # -----------------------------------------------------
    # 2. Asset profile alias match
    # -----------------------------------------------------

    for keyword, profile in profiles.items():

        # Profile exists but there is no local data.
        if keyword not in available_keywords:
            continue

        aliases = profile.get(
            "aliases",
            []
        )

        candidates = [
            keyword,
            *aliases
        ]

        for candidate in candidates:

            if (
                normalize_text(candidate)
                == normalized_subject
            ):
                return keyword

    # -----------------------------------------------------
    # Unsupported subject
    # -----------------------------------------------------

    return None


# =========================================================
# RAW CONTEXT VALIDATION
# =========================================================

def validate_raw_context(
    parsed: dict
) -> dict:

    if not isinstance(
        parsed,
        dict
    ):
        raise ValueError(
            "LLM response JSON object olmalı."
        )

    # -----------------------------------------------------
    # SUBJECT
    # -----------------------------------------------------

    subject = parsed.get(
        "subject"
    )

    if (
        subject is not None
        and not isinstance(subject, str)
    ):
        raise ValueError(
            "subject string veya null olmalı."
        )

    if isinstance(
        subject,
        str
    ):
        subject = subject.strip()

        if not subject:
            subject = None

    # -----------------------------------------------------
    # INTENTS
    # -----------------------------------------------------

    intents = parsed.get(
        "intents",
        []
    )

    if not isinstance(
        intents,
        list
    ):
        raise ValueError(
            "intents list olmalı."
        )

    cleaned_intents = []

    for intent in intents:

        if not isinstance(
            intent,
            str
        ):
            raise ValueError(
                "Her intent string olmalı."
            )

        intent = intent.strip().lower()

        if intent not in ALLOWED_INTENTS:
            raise ValueError(
                f"Unknown intent: {intent}"
            )

        if intent not in cleaned_intents:
            cleaned_intents.append(
                intent
            )

    # -----------------------------------------------------
    # TIME
    # -----------------------------------------------------

    time_context = parsed.get(
        "time",
        []
    )

    if not isinstance(
        time_context,
        list
    ):
        raise ValueError(
            "time list olmalı."
        )

    for item in time_context:

        if not isinstance(
            item,
            dict
        ):
            raise ValueError(
                "time içindeki her eleman "
                "object olmalı."
            )

    # -----------------------------------------------------
    # FOCUS
    # -----------------------------------------------------

    focus = parsed.get(
        "focus",
        []
    )

    if not isinstance(
        focus,
        list
    ):
        raise ValueError(
            "focus list olmalı."
        )

    cleaned_focus = []

    for item in focus:

        if not isinstance(
            item,
            str
        ):
            raise ValueError(
                "focus içindeki her değer "
                "string olmalı."
            )

        item = item.strip().lower()

        if item and item not in cleaned_focus:
            cleaned_focus.append(
                item
            )

    # -----------------------------------------------------
    # COMPARISON
    # -----------------------------------------------------

    comparison = parsed.get(
        "comparison"
    )

    if (
        comparison is not None
        and not isinstance(comparison, str)
    ):
        raise ValueError(
            "comparison string veya null olmalı."
        )

    if isinstance(
        comparison,
        str
    ):
        comparison = comparison.strip()

        if not comparison:
            comparison = None

    return {
        "subject": subject,
        "intents": cleaned_intents,
        "time": time_context,
        "focus": cleaned_focus,
        "comparison": comparison
    }


# =========================================================
# RAW LLM PARSER
# =========================================================

def parse_raw_context(
    user_text: str
) -> dict:

    system_prompt = load_context_prompt()

    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_text
            }
        ]
    )

    content = response.output_text.strip()

    try:

        parsed = json.loads(
            content
        )

    except json.JSONDecodeError:

        raise ValueError(
            "LLM valid JSON döndürmedi:\n"
            f"{content}"
        )

    return validate_raw_context(
        parsed
    )


# =========================================================
# CONTEXT ENRICHMENT
# =========================================================

def enrich_context(
    raw_context: dict
) -> dict:

    available_keywords = (
        get_available_keywords()
    )

    profiles = (
        load_asset_profiles()
    )

    subject = raw_context[
        "subject"
    ]

    keyword = resolve_keyword(
        subject=subject,
        available_keywords=available_keywords,
        profiles=profiles
    )

    asset_type = None
    analysis_dimensions = []

    if keyword is not None:

        profile = profiles.get(
            keyword,
            {}
        )

        asset_type = profile.get(
            "asset_type"
        )

        analysis_dimensions = profile.get(
            "analysis_dimensions",
            []
        )

    return {
        "subject": subject,
        "keyword": keyword,
        "asset_type": asset_type,
        "intents": raw_context["intents"],
        "time": raw_context["time"],
        "focus": raw_context["focus"],
        "analysis_dimensions": analysis_dimensions,
        "comparison": raw_context["comparison"]
    }


# =========================================================
# PUBLIC PARSER
# =========================================================

def parse_input(
    user_text: str
) -> dict:

    raw_context = parse_raw_context(
        user_text
    )

    final_context = enrich_context(
        raw_context
    )

    return final_context


# =========================================================
# TEST QUESTION LOADER
# =========================================================

def load_test_questions() -> list[str]:

    if not QUESTION_FILE.exists():
        raise FileNotFoundError(
            f"Question file bulunamadı: "
            f"{QUESTION_FILE}"
        )

    content = QUESTION_FILE.read_text(
        encoding="utf-8"
    )

    try:

        tree = ast.parse(
            content
        )

    except SyntaxError as e:

        raise ValueError(
            "question_input_parser.txt "
            "valid Python syntax değil.\n"
            f"{e}"
        )

    for node in tree.body:

        if not isinstance(
            node,
            ast.Assign
        ):
            continue

        for target in node.targets:

            if (
                isinstance(target, ast.Name)
                and target.id == "questions"
            ):

                questions = ast.literal_eval(
                    node.value
                )

                if not isinstance(
                    questions,
                    list
                ):
                    raise ValueError(
                        "questions bir list olmalı."
                    )

                if not all(
                    isinstance(question, str)
                    for question in questions
                ):
                    raise ValueError(
                        "questions içindeki tüm "
                        "elemanlar string olmalı."
                    )

                return questions

    raise ValueError(
        "question_input_parser.txt içinde "
        "questions listesi bulunamadı."
    )


# =========================================================
# RESOLUTION STATUS
# =========================================================

def get_resolution_status(
    context: dict
) -> str:

    if context["subject"] is None:
        return "MISSING_SUBJECT"

    if context["keyword"] is None:
        return "UNSUPPORTED"

    return "SUPPORTED"


# =========================================================
# TEST RUNNER
# =========================================================

if __name__ == "__main__":

    questions = load_test_questions()

    results = []

    success_count = 0
    error_count = 0

    supported_count = 0
    unsupported_count = 0
    missing_subject_count = 0

    print()
    print("=" * 80)
    print("CONTEXT PARSER V2 TEST STARTED")
    print("=" * 80)
    print(f"Total questions: {len(questions)}")
    print()

    for index, question in enumerate(
        questions,
        start=1
    ):

        print(
            f"[{index}/{len(questions)}] "
            f"{question}"
        )

        try:

            parsed = parse_input(
                question
            )

            resolution = (
                get_resolution_status(
                    parsed
                )
            )

            if resolution == "SUPPORTED":
                supported_count += 1

            elif resolution == "UNSUPPORTED":
                unsupported_count += 1

            elif resolution == "MISSING_SUBJECT":
                missing_subject_count += 1

            results.append(
                {
                    "test_number": index,
                    "question": question,
                    "status": "SUCCESS",
                    "resolution": resolution,
                    "context": parsed
                }
            )

            success_count += 1

        except Exception as e:

            results.append(
                {
                    "test_number": index,
                    "question": question,
                    "status": "ERROR",
                    "error_type": type(e).__name__,
                    "error": str(e)
                }
            )

            error_count += 1

    # =====================================================
    # WRITE RESULTS
    # =====================================================

    with TEST_RESULT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "=" * 80 + "\n"
        )

        file.write(
            "CONTEXT PARSER V2 TEST RESULTS\n"
        )

        file.write(
            "=" * 80 + "\n\n"
        )

        file.write(
            f"TOTAL TESTS      : {len(questions)}\n"
        )

        file.write(
            f"SUCCESS          : {success_count}\n"
        )

        file.write(
            f"ERROR            : {error_count}\n"
        )

        file.write(
            f"SUPPORTED        : {supported_count}\n"
        )

        file.write(
            f"UNSUPPORTED      : {unsupported_count}\n"
        )

        file.write(
            f"MISSING SUBJECT  : {missing_subject_count}\n"
        )

        file.write(
            "\n"
        )

        for result in results:

            file.write(
                "=" * 80 + "\n"
            )

            file.write(
                f"TEST {result['test_number']}\n"
            )

            file.write(
                "=" * 80 + "\n\n"
            )

            file.write(
                "QUESTION:\n"
            )

            file.write(
                result["question"]
                + "\n\n"
            )

            file.write(
                "STATUS:\n"
            )

            file.write(
                result["status"]
                + "\n\n"
            )

            if (
                result["status"]
                == "SUCCESS"
            ):

                file.write(
                    "RESOLUTION:\n"
                )

                file.write(
                    result["resolution"]
                    + "\n\n"
                )

                file.write(
                    "PARSED CONTEXT:\n"
                )

                file.write(
                    json.dumps(
                        result["context"],
                        indent=2,
                        ensure_ascii=False
                    )
                )

                file.write(
                    "\n\n"
                )

            else:

                file.write(
                    "ERROR TYPE:\n"
                )

                file.write(
                    result["error_type"]
                    + "\n\n"
                )

                file.write(
                    "ERROR:\n"
                )

                file.write(
                    result["error"]
                    + "\n\n"
                )

        file.write(
            "=" * 80 + "\n"
        )

        file.write(
            "SUMMARY\n"
        )

        file.write(
            "=" * 80 + "\n\n"
        )

        file.write(
            f"TOTAL TESTS      : {len(questions)}\n"
        )

        file.write(
            f"SUCCESS          : {success_count}\n"
        )

        file.write(
            f"ERROR            : {error_count}\n"
        )

        file.write(
            f"SUPPORTED        : {supported_count}\n"
        )

        file.write(
            f"UNSUPPORTED      : {unsupported_count}\n"
        )

        file.write(
            f"MISSING SUBJECT  : {missing_subject_count}\n"
        )

    # =====================================================
    # TERMINAL SUMMARY
    # =====================================================

    print()
    print("=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)

    print(
        f"Total           : {len(questions)}"
    )

    print(
        f"Success         : {success_count}"
    )

    print(
        f"Error           : {error_count}"
    )

    print(
        f"Supported       : {supported_count}"
    )

    print(
        f"Unsupported     : {unsupported_count}"
    )

    print(
        f"Missing subject : {missing_subject_count}"
    )

    print(
        f"Output          : {TEST_RESULT_FILE}"
    )