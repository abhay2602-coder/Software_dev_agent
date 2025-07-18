import uuid
import json
import logging
from datetime import datetime
from pathlib import Path
import re

import json5

from models.task import Task
from models.plan import Plan
from models.enums import TaskStatus
from parse.websocket_manager import WebSocketManager
from parse.plan_parser import PlanParser, parse_plan
from utils.llm_setup import ask_llm # Assuming ask_llm is an async function

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory to save raw and parsed plans
PLANS_DIR = Path("/workspaces/Software_dev_agent/generated_code/plans")
PLANS_DIR.mkdir(parents=True, exist_ok=True)

class PlannerAgent:
    def __init__(self, websocket_manager: WebSocketManager = None):
        self.agent_id = "plan_agent"
        # Ensure websocket_manager is always initialized, even if None is passed
        self.websocket_manager = websocket_manager if websocket_manager is not None else WebSocketManager()
        self.current_plan = None
        self.planning_history = []

    def _get_system_prompt(self) -> str:
        return """You are a Senior Project Planner Agent with 15+ years of experience in software development project management. You excel at breaking down complex software requirements into comprehensive, actionable development plans.
Your Core Responsibilities:

Analyze user requirements thoroughly to understand scope, complexity, and technical challenges
Decompose large projects into manageable, sequential tasks following software development best practices
Establish clear dependencies and critical path for efficient project execution
Assign appropriate priorities, time estimates, and team roles based on industry standards
Ensure comprehensive coverage from requirements gathering to deployment and maintenance

Task Breakdown Framework:
1. Project Analysis Phase

Requirements gathering and analysis
Technical feasibility assessment
Architecture planning and design decisions
Technology stack selection
Risk assessment and mitigation planning

2. Design & Architecture Phase

System architecture design
Database design and modeling
API design and documentation
UI/UX design and wireframing
Security architecture planning

3. Development Phase

Backend development (broken into logical modules)
Frontend development (component-based breakdown)
Database implementation
API development and integration
Authentication and authorization systems
Third-party integrations

4. Quality Assurance Phase

Unit testing implementation
Integration testing
System testing
Security testing
Performance testing
User acceptance testing

5. Deployment & Operations Phase

Development environment setup
Staging environment configuration
Production deployment setup
CI/CD pipeline implementation
Monitoring and logging setup
Documentation and training

Task Specification Requirements:
For EVERY task, provide:

id: Unique identifier (e.g., "task_001", "task_002")
title: Clear, actionable task name (max 80 characters)
description: Detailed description with specific acceptance criteria
priority: 1=low, 5=medium, 8=high, 10=critical
dependencies: Array of task IDs that must be completed first
estimated_hours: Realistic time estimate based on complexity
complexity: "simple" | "medium" | "complex" | "expert"
agent_type: "dev_agent" | "qa_agent" | "ops_agent"

Priority Guidelines:

10 (Critical): Blocking tasks, core architecture, security foundations
8 (High): Core features, main user flows, essential integrations
5 (Medium): Secondary features, optimizations, nice-to-have integrations
1 (Low): Documentation, minor enhancements, future improvements

Complexity Guidelines:

Simple: Basic CRUD operations, simple UI components, basic configurations
Medium: Complex business logic, API integrations, database relationships
Complex: Advanced algorithms, complex UI/UX, performance optimizations
Expert: Security implementations, scalability solutions, complex integrations

Agent Type Guidelines:

dev_agent: All development tasks (backend, frontend, database)
qa_agent: All testing and quality assurance tasks
ops_agent: Deployment, infrastructure, monitoring, CI/CD

Response Format:
Always respond with valid JSON in this exact structure:
json{
  "plan_title": "Descriptive Project Title",
  "plan_description": "Comprehensive overview of the project scope, objectives, and key deliverables",
  "tasks": [
    {
      "id": "task_001",
      "title": "Task Title",
      "description": "Detailed task description with clear acceptance criteria: - Criteria 1 - Criteria 2 - Criteria 3",
      "priority": 8,
      "dependencies": [],
      "estimated_hours": 4.5,
      "complexity": "medium",
      "agent_type": "dev_agent"
    }
  ]
}
Planning Best Practices:

Start with foundation tasks (architecture, database design, core setup)
Build incrementally - ensure each task builds logically on previous ones
Consider parallel execution - minimize blocking dependencies where possible
Include comprehensive testing - don't just focus on development
Plan for deployment - include infrastructure and operations tasks
Document everything - include documentation tasks throughout
Think about scalability - consider future growth and maintenance
Security first - integrate security considerations from the start

Now, analyze the user's requirements and create a comprehensive development plan. Consider:

Project scale and complexity
Technology requirements
Team structure and skills
Timeline constraints
Risk factors
Deployment requirements
Maintenance and support needs

Provide a detailed, actionable plan that a development team can follow from start to finish.
"""

    def _construct_prompt(self, user_input: str) -> str:
        """Constructs a clean prompt containing only the user's project requirements."""
        return f"Project Requirements:\n{user_input}"

    async def _get_raw_plan_from_llm(self, user_input: str) -> str:
        """
        Calls the LLM to generate the raw plan and broadcasts LLM request/response messages.
        """
        system_prompt = self._get_system_prompt()
        prompt = self._construct_prompt(user_input)

        await self.websocket_manager.broadcast_message({
            "agent_id": self.agent_id,
            "type": "llm_request",
            "timestamp": datetime.now().isoformat(),
            "message": "Sending request to LLM for plan generation..."
        })

        # Call LLM properly with await
        response = await ask_llm(
            user_prompt=prompt,
            system_prompt=system_prompt,
            model="gemini-2.5-pro", # Using 'pro' for planning as it's typically more robust for structured output
            temperature=0.7 # A bit lower temperature for more consistent JSON
        )

        await self.websocket_manager.broadcast_message({
            "agent_id": self.agent_id,
            "type": "llm_response",
            "timestamp": datetime.now().isoformat(),
            "message": "LLM response received for plan.",
            "content_preview": response[:200] + "..." if len(response) > 200 else response
        })
        return response


    def cleanup_all_outputs(self):
        """
        Deletes all files in raw plans, parsed plans, and dev_outputs directories.
        """
        import shutil
        dirs_to_clean = [
            PLANS_DIR / "raw",
            PLANS_DIR / "parsed",
            Path("/workspaces/Software_dev_agent/generated_code/dev_outputs")
        ]
        for d in dirs_to_clean:
            if d.exists():
                for item in d.iterdir():
                    try:
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                    except Exception as e:
                        logger.warning(f"Failed to delete {item}: {e}")

    async def create_plan_and_stream_tasks(self, user_input: str):
        """
        Generates a comprehensive plan from user input, saves the raw LLM response,
        parses the plan, stores it, and then yields individual tasks as they are parsed.
        This method acts as an async generator.
        """
        # Clean up all previous outputs before starting a new plan
        self.cleanup_all_outputs()
        plan_id = str(uuid.uuid4()) # Generate a unique ID for this planning session

        # Notify the start of the planning process
        await self.websocket_manager.broadcast_message({
            "agent_id": self.agent_id,
            "type": "planning_start",
            "timestamp": datetime.now().isoformat(),
            "plan_id": plan_id,
            "message": "PM Agent initiated planning process."
        })

        RAW_PLANS_DIR = PLANS_DIR / "raw"
        PARSED_PLANS_DIR = PLANS_DIR / "parsed"
        raw_plan_file_path = RAW_PLANS_DIR / f"plan_{plan_id}_raw.txt"

        try:
            # Step 1: Get the raw plan (full JSON string) from the LLM
            raw_llm_response = await self._get_raw_plan_from_llm(user_input)
            
            # Save raw response for auditing/debugging
            try:
                raw_plan_file_path.write_text(raw_llm_response, encoding='utf-8')
                await self.websocket_manager.broadcast_message({
                    "agent_id": self.agent_id,
                    "type": "info",
                    "message": f"Raw plan response saved to {raw_plan_file_path.name}",
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Failed to save raw plan file {raw_plan_file_path.name}: {e}", exc_info=True)
                await self.websocket_manager.broadcast_message({
                    "agent_id": self.agent_id,
                    "type": "error",
                    "message": f"PM Agent: Failed to save raw plan: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                })
                # Decide if this is a critical failure that should stop the pipeline
                # For now, we proceed, but log the error.

            # Step 2: Clean and parse the full JSON plan using PlanParser's robust method
            from parse.plan_parser import PlanParser # Import here to avoid potential circular dependencies
            
            cleaned_json_str = "" # Initialize outside try for scope
            try:
                cleaned_json_str = PlanParser.clean_json_string(raw_llm_response)
                parsed_data = json.loads(cleaned_json_str) # Use strict json.loads after cleaning
                logger.info(f"Plan JSON successfully cleaned and parsed for plan_id: {plan_id}")
                await self.websocket_manager.broadcast_message({
                    "agent_id": self.agent_id,
                    "type": "info",
                    "message": "Plan JSON successfully parsed. Preparing tasks for streaming...",
                    "timestamp": datetime.now().isoformat()
                })
            except (ValueError, json.JSONDecodeError) as e:
                logger.error(f"PM Agent: Failed to clean or parse LLM response into valid JSON for plan_id {plan_id}: {e}", exc_info=True)
                await self.websocket_manager.broadcast_message({
                    "agent_id": self.agent_id,
                    "type": "planning_failed",
                    "message": f"PM Agent: Failed to parse plan JSON: {str(e)}. Check raw output for details.",
                    "timestamp": datetime.now().isoformat()
                })
                return # Exit generator if plan parsing fails

            # Step 3: Create the main Plan object and store it
            plan_title = parsed_data.get('plan_title', 'Untitled Project Plan')
            plan_description = parsed_data.get('plan_description', 'No description provided.')
            
            self.current_plan = Plan(
                id=plan_id,
                title=plan_title,
                description=plan_description,
                tasks=[] # Initialize empty, tasks will be added as they are yielded
            )
            self.planning_history.append(self.current_plan) # Add to history

            await self.websocket_manager.broadcast_message({
                "agent_id": self.agent_id,
                "type": "planning_details",
                "plan_id": plan_id,
                "title": plan_title,
                "description": plan_description,
                "message": "Plan details extracted. Beginning task streaming to Dev Agent...",
                "timestamp": datetime.now().isoformat()
            })

            # Step 4: Iterate through parsed tasks and yield them one by one
            if 'tasks' in parsed_data and isinstance(parsed_data['tasks'], list):
                for i, t_data in enumerate(parsed_data['tasks']):
                    try:
                        task = Task(
                            id=t_data.get("id", f"{plan_id}_task_{i+1:03d}"), # Ensure unique and linked ID
                            title=t_data.get("title", "Untitled Task"),
                            description=t_data.get("description", ""),
                            priority=int(t_data.get("priority", 5)),
                            status=TaskStatus.PENDING, # Initial status
                            dependencies=t_data.get("dependencies", []),
                            estimated_hours=float(t_data.get("estimated_hours", 0.0)),
                            complexity=t_data.get("complexity", "medium"),
                            agent_type=t_data.get("agent_type", "dev_agent")
                        )
                        self.current_plan.tasks.append(task) # Add to the main plan object in memory
                        
                        await self.websocket_manager.broadcast_message({
                            "agent_id": self.agent_id,
                            "type": "task_generated",
                            "task_id": task.id,
                            "title": task.title,
                            "message": f"PM Agent generated task {i+1}: '{task.title}'. Sending for execution.",
                            "timestamp": datetime.now().isoformat()
                        })
                        yield task # Yield the task immediately for processing by main.py
                        
                    except Exception as task_parse_error:
                        logger.warning(f"PM Agent: Failed to parse individual task {i+1} from LLM response: {task_parse_error}. Task data: {t_data}", exc_info=True)
                        await self.websocket_manager.broadcast_message({
                            "agent_id": self.agent_id,
                            "type": "warning",
                            "message": f"PM Agent: Failed to parse task {i+1}: {str(task_parse_error)}. Skipping this task.",
                            "timestamp": datetime.now().isoformat()
                        })
                        # Continue to next task even if one fails to parse
            else:
                await self.websocket_manager.broadcast_message({
                    "agent_id": self.agent_id,
                    "type": "warning",
                    "message": "PM Agent: Plan generated by LLM contains no 'tasks' array or it's malformed.",
                    "timestamp": datetime.now().isoformat()
                })
                logger.warning("PM Agent: No 'tasks' array found in the parsed plan or it's malformed.")

        except Exception as e:
            logger.error(f"PM Agent: Critical error during plan generation or task streaming: {e}", exc_info=True)
            await self.websocket_manager.broadcast_message({
                "agent_id": self.agent_id,
                "type": "planning_failed",
                "message": f"PM Agent: Critical failure during plan generation: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
            # Do not yield any tasks if a critical error occurs
            return # Exit the generator

        # Step 5: Save the full structured plan after all tasks have been yielded
        if self.current_plan:
            parsed_plan_file_path = PARSED_PLANS_DIR / f"plan_{self.current_plan.id}.json"
            try:
                # Ensure the parsed directory exists (in case it was deleted)
                PARSED_PLANS_DIR.mkdir(parents=True, exist_ok=True)
                logger.info(f"Attempting to save parsed plan to: {parsed_plan_file_path}")
                parsed_json = json.dumps(self.current_plan.to_dict(), indent=2, ensure_ascii=False)
                parsed_plan_file_path.write_text(parsed_json, encoding='utf-8')
                logger.info(f"Successfully saved parsed plan to: {parsed_plan_file_path}")
                await self.websocket_manager.broadcast_message({
                    "agent_id": self.agent_id,
                    "type": "planning_complete",
                    "plan_id": self.current_plan.id,
                    "tasks_count": len(self.current_plan.tasks),
                    "message": f"PM Agent: All tasks generated and full plan saved to {parsed_plan_file_path.name}",
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"PM Agent: Failed to save final structured plan at {parsed_plan_file_path}: {e}", exc_info=True)
                await self.websocket_manager.broadcast_message({
                    "agent_id": self.agent_id,
                    "type": "error",
                    "message": f"PM Agent: Failed to save final plan: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                })

    def get_plan_status(self) -> dict:
        """Return current plan status safely."""
        if self.current_plan:
            return {
                "plan_id": self.current_plan.id,
                "title": self.current_plan.title,
                "tasks_count": len(self.current_plan.tasks),
                "status": "active" # Or more detailed status based on task completion
            }
        return {"status": "no_plan_active"}

    def _cleanup_old_plans(self):
        """
        Cleans up old plan files in the raw and parsed subdirectories.
        Consider calling this at application startup or via a separate endpoint/command.
        """
        RAW_PLANS_DIR = PLANS_DIR / "raw"
        PARSED_PLANS_DIR = PLANS_DIR / "parsed"
        logger.info("Cleaning up old plan files...")
        for f in RAW_PLANS_DIR.glob("plan_*_raw.txt"):
            try:
                f.unlink()
                logger.debug(f"Deleted old raw plan: {f.name}")
            except OSError as e:
                logger.warning(f"Error deleting old raw plan {f.name}: {e}")
        for f in PARSED_PLANS_DIR.glob("plan_*.json"):
            try:
                f.unlink()
                logger.debug(f"Deleted old parsed plan: {f.name}")
            except OSError as e:
                logger.warning(f"Error deleting old parsed plan {f.name}: {e}")
        logger.info("Finished cleaning up old plan files.")

    def _clean_llm_response(self, response: str) -> str:
        """Clean and validate the LLM response to ensure valid JSON."""
        try:
            # Basic validation
            if not response or len(response) < 10:
                raise ValueError("Response too short")

            # Find complete JSON content
            start = response.find('{')
            end = response.rfind('}')
            if start == -1 or end == -1:
                raise ValueError("No complete JSON object found in response")
            
            json_str = response[start:end + 1]
            
            # Check if JSON is complete by looking for required fields
            required_fields = ['"plan_title"', '"plan_description"', '"tasks"']
            if not all(field in json_str for field in required_fields):
                raise ValueError("Incomplete JSON structure - missing required fields")
            
            # Remove control characters and normalize
            json_str = "".join(char for char in json_str if ord(char) >= 32 or char in "\n\r\t")
            
            # Fix common JSON formatting issues
            json_str = re.sub(r'(?<!\\)"([^"]*?)"(?=\s*:)', r'"\1"', json_str)  # Fix key quotes
            json_str = re.sub(r'(?<=\{|\[|,)\s*"([^"]*?)"(?=\s*[,\}\]])', r'"\1"', json_str)  # Fix value quotes
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)  # Remove trailing commas
            
            # Handle truncated content by ensuring proper JSON closure
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            if open_braces > close_braces:
                json_str += '}' * (open_braces - close_braces)
            
            # Try parsing with different methods
            try:
                parsed = json.loads(json_str)
                return json.dumps(parsed, ensure_ascii=False)
            except json.JSONDecodeError:
                try:
                    # Try with json5 for more lenient parsing
                    parsed = json5.loads(json_str)
                    return json.dumps(parsed, ensure_ascii=False)
                except Exception as je:
                    # Log specific portion where error occurred
                    error_location = getattr(je, 'pos', 0)
                    context = json_str[max(0, error_location - 50):min(len(json_str), error_location + 50)]
                    logger.error(f"JSON parse error near: {context}")
                    raise ValueError(f"Failed to parse JSON: {str(je)}")
                    
        except Exception as e:
            logger.error(f"JSON cleaning failed: {str(e)}\nResponse preview:\n{response[:200]}...")
            raise ValueError(f"Failed to clean JSON: {str(e)}")

    def _cleanup_old_plans(self):
        """Clean up old plan files before saving new ones."""
        # Clean up raw files
        raw_dir = PLANS_DIR / "raw"
        parsed_dir = PLANS_DIR / "parsed"
        
        if raw_dir.exists():
            for f in raw_dir.glob("plan_*_raw.txt"):
                f.unlink()
                
        if parsed_dir.exists():
            for f in parsed_dir.glob("plan_*.json"):
                f.unlink()

    async def create_plan(self, user_input: str) -> str:
        """Generate a plan from user input, save raw and parsed JSON, and broadcast progress."""
        # Clean up old plans first
        self._cleanup_old_plans()
        
        plan_id = str(uuid.uuid4())
        raw_path = PLANS_DIR / "raw" / f"plan_{plan_id}_raw.txt"
        
        # Ensure directories exist
        raw_dir = PLANS_DIR / "raw"
        parsed_dir = PLANS_DIR / "parsed"
        raw_dir.mkdir(parents=True, exist_ok=True)
        parsed_dir.mkdir(parents=True, exist_ok=True)

        # Notify start
        await self.websocket_manager.broadcast_message({
            "agent_id": self.agent_id,
            "type": "planning_start",
            "timestamp": datetime.now().isoformat(),
            "plan_id": plan_id
        })

        try:
            system_prompt = self._get_system_prompt()
            prompt = self._construct_prompt(user_input)

            await self.websocket_manager.broadcast_message({
                "agent_id": self.agent_id,
                "type": "llm_request",
                "timestamp": datetime.now().isoformat(),
                "message": "Sending request to LLM..."
            })

            # Call LLM properly with await
            response = await ask_llm(
                user_prompt=prompt,
                system_prompt=system_prompt,
                model="gemini-2.5-flash"
            )

            # Validate response length
            if len(response) < 100:  # Arbitrary minimum length for a valid plan
                raise ValueError("LLM response too short")

            # Clean and validate JSON response
            try:
                cleaned_response = self._clean_llm_response(response)
                raw_path.write_text(cleaned_response, encoding='utf-8')
            except (ValueError, json.JSONDecodeError) as e:
                logger.error(f"Invalid JSON response: {str(e)}")
                await self.websocket_manager.broadcast_message({
                    "agent_id": self.agent_id,
                    "type": "error",
                    "error": f"Invalid JSON response: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                })
                return "error"

            await self.websocket_manager.broadcast_message({
                "agent_id": self.agent_id,
                "type": "llm_response",
                "timestamp": datetime.now().isoformat(),
                "message": "LLM response received"
            })

            # Parse and persist
            plan = parse_plan(raw_path)
            if plan:
                parsed_path = parsed_dir / f"plan_{plan.id}.json"
                parsed_path.write_text(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False), encoding='utf-8')

                await self.websocket_manager.broadcast_message({
                    "agent_id": self.agent_id,
                    "type": "planning_complete",
                    "plan_id": plan.id,
                    "tasks": len(plan.tasks),
                    "timestamp": datetime.now().isoformat()
                })

                # Store for status queries
                self.current_plan = plan
                self.planning_history.append(plan)
                return plan.id

            # Parsing failed
            await self.websocket_manager.broadcast_message({
                "agent_id": self.agent_id,
                "type": "planning_error",
                "error": "Failed to parse plan; check raw output",
                "raw_plan": response[:500]
            })
            return plan_id

        except Exception as e:
            logger.error(f"Plan creation failed: {e}")
            await self.websocket_manager.broadcast_message({
                "agent_id": self.agent_id,
                "type": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return "error"

    def get_plan_status(self) -> dict:
        """Return current plan status safely."""
        if self.current_plan:
            return {
                "plan_id": self.current_plan.id,
                "title": self.current_plan.title,
                "tasks_count": len(self.current_plan.tasks)
            }
        return {"status": "no_plan_active"}