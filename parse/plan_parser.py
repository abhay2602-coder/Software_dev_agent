import json
import re
import json5 # Used for more lenient JSON parsing
import logging
from pathlib import Path
from typing import Optional
from models.plan import Plan
from models.task import Task
from models.enums import TaskStatus

logger = logging.getLogger(__name__)

# Ensure this path is correct and accessible
RAW_PLAN_DIR = Path("/workspaces/Software_Developer_AgenticAI/generated_code/plans")
RAW_PLAN_DIR.mkdir(parents=True, exist_ok=True) # Ensure directory exists

class PlanParser:
    """Parent class for plan parsing utilities."""

    @staticmethod
    def clean_json_string(text: str) -> str:
        """
        Clean and normalize LLM response to make it valid JSON.
        This version is more aggressive in trying to isolate a single JSON object.
        """
        original_text_preview = text[:500].replace('\n', '\\n') + ('...' if len(text) > 500 else '')
        logger.debug(f"Attempting to clean JSON. Original text start: {original_text_preview}")

        # Step 1: Aggressively try to find a JSON object enclosed by `{...}`
        # This regex looks for the first '{' and tries to match until the corresponding '}'
        # and then ensures there isn't significant non-whitespace data after it.
        # This is tricky because JSON can be nested, but for a single root object, it's a good start.
        match = re.search(r'\{.*\}', text, re.DOTALL)
        json_str = ""
        if match:
            json_str = match.group(0)
            logger.debug(f"Regex found potential JSON block. Start: {json_str[:200].replace('\n', '\\n')}...")
        else:
            # If no full {..} block found, try to extract between first { and last }
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and start < end:
                json_str = text[start:end + 1]
                logger.debug(f"Fallback: Extracted from first '{{' to last '}}'. Start: {json_str[:200].replace('\n', '\\n')}...")
            else:
                raise ValueError("No complete JSON object-like structure found in response.")

        # Step 2: Remove common LLM markdown artifacts (though we prompt against them)
        json_str = re.sub(r'```(?:json)?\s*', '', json_str, flags=re.IGNORECASE).strip()
        json_str = re.sub(r'\s*```\s*$', '', json_str).strip() # Remove closing fence

        # Step 3: Basic syntactic cleanup that json5 can handle, but helps ensure validity
        # Replace single quotes with double quotes for keys/strings
        # This is a bit risky if single quotes are legitimately inside a string value
        # but common for LLMs hallucinating JS-like objects.
        # json_str = json_str.replace("'", '"') # <-- Commenting out, json5 handles this better if it's strictly object keys

        # Remove trailing commas before a closing bracket or brace. json5 handles this, but good to normalize.
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

        # Normalize invisible characters, remove anything below ASCII 32 except newlines/tabs
        json_str = "".join(c for c in json_str if ord(c) >= 32 or c in "\n\r\t")

        # Step 4: Bracket balancing (a last-ditch effort, doesn't fix deep structural errors)
        # Only add closing brackets/braces if the count is off and it's a simple case.
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)
            logger.warning(f"Added {open_braces - close_braces} closing curly braces for balancing.")
        
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        if open_brackets > close_brackets:
            json_str += ']' * (open_brackets - close_brackets)
            logger.warning(f"Added {open_brackets - close_brackets} closing square brackets for balancing.")

        logger.debug(f"After initial cleaning and balancing: {json_str[:200].replace('\n', '\\n')}...")

        # Step 5: Final parsing attempt with json5 first, then standard json
        try:
            # Use json5 for more relaxed parsing first
            parsed_data = json5.loads(json_str)
            logger.info("Successfully parsed with json5.")
        except json.JSONDecodeError as json_err:
            logger.warning(f"json5 failed, trying json.loads. Error: {json_err}")
            try:
                parsed_data = json.loads(json_str)
                logger.info("Successfully parsed with standard json.")
            except json.JSONDecodeError as standard_json_err:
                logger.error(f"Failed with both json5 and json.loads. Standard JSON Error: {standard_json_err}")
                raise standard_json_err # Re-raise the original JSON error
        except Exception as e:
            logger.error(f"Unexpected error during final parsing: {e}")
            raise e # Re-raise other exceptions

        # Convert back to strict JSON format for consistent output, ensure_ascii=False for non-ASCII chars
        final_json_str = json.dumps(parsed_data, ensure_ascii=False)
        logger.debug(f"Final cleaned JSON: {final_json_str[:200].replace('\n', '\\n')}...")
        return final_json_str

    @staticmethod
    def parse_plan_file(file_path: Path) -> Optional[Plan]:
        """Read raw plan file and return structured Plan object."""
        try:
            raw_text = file_path.read_text(encoding="utf-8")
            cleaned_json = PlanParser.clean_json_string(raw_text)

            # Already loaded by json5/json.loads in clean_json_string, so just parse it again from string
            # This ensures consistent loading regardless of which parser succeeded in cleaning
            plan_data = json.loads(cleaned_json) # Always load with strict json.loads after cleaning

            # Validate minimal fields
            if not all(k in plan_data for k in ['plan_title', 'plan_description', 'tasks']):
                raise ValueError("Missing one of: plan_title, plan_description, tasks in parsed plan data.")

            # Parse tasks
            tasks = []
            for i, t in enumerate(plan_data['tasks'], 1):
                try:
                    task = Task(
                        id=t.get("id", f"task_{i:03d}"), # Ensure 'id' is always present, use formatted default
                        title=t.get("title", "Untitled Task"),
                        description=t.get("description", ""),
                        priority=int(t.get("priority", 5)),
                        status=TaskStatus.PENDING,
                        dependencies=t.get("dependencies", []),
                        estimated_hours=float(t.get("estimated_hours", 0.0)), # Default to float 0.0
                        complexity=t.get("complexity", "medium"),
                        agent_type=t.get("agent_type", "dev_agent")
                    )
                    tasks.append(task)
                except Exception as e:
                    # Log the problematic task data for debugging
                    logger.warning(f"Failed to parse task {i} due to: {e}. Task data: {t}")
                    # Decide if you want to skip this task or fail the whole plan
                    # For now, we continue but warn. If critical, re-raise.

            if not tasks and plan_data['tasks']: # If plan_data['tasks'] had items but none were valid
                 logger.warning("No valid tasks could be parsed from the LLM response, even though tasks array was present.")
                 # Optionally raise an error here if a plan must have tasks
                 # raise ValueError("No valid tasks could be parsed from the LLM response.")

            # if not tasks: # This check is now covered by the previous logic
            #     raise ValueError("No valid tasks found")

            # Extract UUID from file name for plan ID
            plan_id_from_file = file_path.stem.replace("plan_", "").replace("_raw", "")

            return Plan(
                id=plan_id_from_file,
                title=plan_data['plan_title'],
                description=plan_data['plan_description'],
                tasks=tasks
            )

        except Exception as e:
            logger.error(f"Failed to parse plan from {file_path.name}: {e}", exc_info=True) # Add exc_info=True for full traceback
            return None

    @staticmethod
    def parse_all_raw_plans():
        """Parse all *_raw.txt plans in the directory and save them as JSON."""
        # Ensure RAW_PLAN_DIR exists
        RAW_PLAN_DIR.mkdir(parents=True, exist_ok=True)
        
        for file in RAW_PLAN_DIR.glob("plan_*_raw.txt"):
            logger.info(f"Attempting to parse raw plan file: {file.name}")
            plan = PlanParser.parse_plan_file(file)
            if plan:
                output_path = RAW_PLAN_DIR / f"plan_{plan.id}.json"
                try:
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(plan.to_dict(), f, indent=2, ensure_ascii=False)
                    logger.info(f"✅ Parsed and saved: {output_path.name}")
                except Exception as e:
                    logger.error(f"Failed to save parsed plan {output_path.name}: {e}")
            else:
                logger.warning(f"⚠️ Skipped invalid plan file: {file.name}")


def parse_plan(file_path: Path) -> Optional[Plan]:
    """Parse a plan file and return a Plan object (for import convenience)."""
    return PlanParser.parse_plan_file(file_path)