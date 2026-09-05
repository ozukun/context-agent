import ast
import json
import os
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


# =========================================================
# AVAILABLE KEYWORDS
# =========================================================

def get_available_keywords() -> list[str]:

    if not DATA_SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Data source folder bulunamadı: {DATA_SOURCE_DIR}"
        )

    keywords = [
        folder.name
        for folder in DATA_SOURCE_DIR.iterdir()
        if folder.is_dir()
    ]

    return sorted(keywords)


# =========================================================
# LOAD CONTEXT PROMPT
# =========================================================

def load_context_prompt(
    keywords: list[str]
) -> str:

    context_file = (
        CONTEXT_DIR
        / "context_input_parser.txt"
    )

    if not context_file.exists():
        raise FileNotFoundError(
            f"Context file bulunamadı: {context_file}"
        )

    context_template = context_file.read_text(
        encoding="utf-8"
    )

    system_prompt = context_template.replace(
        "{keywords}",
        json.dumps(
            keywords,
            ensure_ascii=False
        )
    )

    return system_prompt


# =========================================================
# LOAD TEST QUESTIONS
# =========================================================

def load_test_questions() -> list[str]:

    question_file = (
        CONTEXT_DIR
        / "question_input_parser.txt"
    )

    if not question_file.exists():
        raise FileNotFoundError(
            f"Question file bulunamadı: {question_file}"
        )

    content = question_file.read_text(
        encoding="utf-8"
    )

    try:
        tree = ast.parse(content)

    except SyntaxError as e:
        raise ValueError(
            "question_input_parser.txt valid Python syntax değil.\n"
            f"{e}"
        )

    for node in tree.body:

        if isinstance(node, ast.Assign):

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
                            "'questions' bir list olmalı."
                        )

                    if not all(
                        isinstance(question, str)
                        for question in questions
                    ):
                        raise ValueError(
                            "questions içindeki tüm elemanlar string olmalı."
                        )

                    return questions

    raise ValueError(
        "question_input_parser.txt içinde "
        "'questions' listesi bulunamadı."
    )


# =========================================================
# PARSE USER INPUT
# =========================================================

def parse_input(
    user_text: str
) -> dict:

    keywords = get_available_keywords()

    system_prompt = load_context_prompt(
        keywords
    )

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

    # =====================================================
    # BASIC VALIDATION
    # =====================================================

    keyword = parsed.get(
        "keyword"
    )

    if keyword not in keywords:

        raise ValueError(
            f"Unknown keyword: {keyword}"
        )

    return parsed


# =========================================================
# TEST RUNNER
# =========================================================

if __name__ == "__main__":

    questions = load_test_questions()

    output_file = (
        CONTEXT_DIR
        / "test_results_input_parser.txt"
    )

    results = []

    success_count = 0
    error_count = 0

    print()
    print("=" * 80)
    print("CONTEXT PARSER TEST STARTED")
    print("=" * 80)

    print(
        f"Total questions: {len(questions)}"
    )

    print()

    # =====================================================
    # RUN TESTS
    # =====================================================

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

            results.append(
                {
                    "test_number": index,
                    "question": question,
                    "status": "SUCCESS",
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
    # WRITE TEST RESULTS
    # =====================================================

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "=" * 80
            + "\n"
        )

        file.write(
            "CONTEXT PARSER TEST RESULTS\n"
        )

        file.write(
            "=" * 80
            + "\n\n"
        )

        file.write(
            f"TOTAL TESTS   : {len(questions)}\n"
        )

        file.write(
            f"SUCCESS       : {success_count}\n"
        )

        file.write(
            f"ERROR         : {error_count}\n"
        )

        file.write(
            "\n"
        )

        # =================================================
        # INDIVIDUAL RESULTS
        # =================================================

        for result in results:

            file.write(
                "=" * 80
                + "\n"
            )

            file.write(
                f"TEST {result['test_number']}\n"
            )

            file.write(
                "=" * 80
                + "\n\n"
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

            # =============================================
            # SUCCESS
            # =============================================

            if (
                result["status"]
                == "SUCCESS"
            ):

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

            # =============================================
            # ERROR
            # =============================================

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


        # =================================================
        # SUMMARY
        # =================================================

        file.write(
            "=" * 80
            + "\n"
        )

        file.write(
            "SUMMARY\n"
        )

        file.write(
            "=" * 80
            + "\n\n"
        )

        file.write(
            f"TOTAL TESTS : {len(questions)}\n"
        )

        file.write(
            f"SUCCESS     : {success_count}\n"
        )

        file.write(
            f"ERROR       : {error_count}\n"
        )


    # =====================================================
    # TERMINAL SUMMARY
    # =====================================================

    print()
    print("=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)

    print(
        f"Total   : {len(questions)}"
    )

    print(
        f"Success : {success_count}"
    )

    print(
        f"Error   : {error_count}"
    )

    print(
        f"Output  : {output_file}"
    )